"""Evaluation entry point: plan with the diffusion policy and roll out in OGBench."""

from dataclasses import asdict
import os
import argparse
import sys
from typing import List, Dict
import math

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import h5py

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(True)

from .policy import CompositionalPolicy
from .dataset.common import EnvSpec, load_env_specs
from .dataset.ogbench import (
    ogb_make_env, ogb_load_train_dataset, make_normalizer,
    ogb_maze_unit, ogb_offset_x, ogb_offset_y,
)
from .utils.io import mkdir, save_json, checkpoint_path, freeze_model
from .utils.maze_path import shortest_path_plan_xy
from .utils.visualization import save_eval_samples_plot, save_eval_tracking_plot

from .planner import build_planner_diffusion_for_env
from .invdyn import build_invdyn_model_for_env
from .ecd_prior import FittedGaussianMarkovPrior


def set_seed(seed: int):
    """Seed numpy and torch (including CUDA) RNGs for reproducible evaluation."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_eval_problems(h5path: str) -> Dict[str, np.ndarray]:
    """Load eval start/goal problems from an HDF5 file into a dict of arrays."""
    with h5py.File(h5path, "r") as f:
        return {k: f[k][:] for k in f.keys()}


def set_start_state(env, start_state: np.ndarray):
    """Set the environment to ``start_state`` using whichever setter the env exposes."""
    obs0 = env.get_ob()
    if hasattr(env, "set_state_with_obs") and (start_state.shape == obs0.shape):
        env.set_state_with_obs(start_state)
        return
    if hasattr(env, "set_state_with_full"):
        env.set_state_with_full(start_state)
        return
    # Some OgBench gym environments (e.g. PointEnv) only expose the Mujoco qpos/qvel
    # state. In that case we can directly call set_state by filling qpos with the
    # provided observation and zeroing the velocity.
    data = getattr(env, "data", None)
    if hasattr(env, "set_state") and data is not None:
        qpos = getattr(data, "qpos", None)
        qvel = getattr(data, "qvel", None)
        if qpos is not None and qvel is not None and qpos.size == start_state.size:
            qpos_copy = qpos.copy()
            qvel_copy = qvel.copy()
            qpos_copy[:] = start_state
            qvel_copy[:] = 0.0
            env.set_state(qpos_copy, qvel_copy)
            return
    if hasattr(env, "set_xy_with_0vel") and start_state.size == 2:
        env.set_xy_with_0vel(start_state)
        return
    raise RuntimeError(f"Cannot set start state: start_state shape={start_state.shape}, env.get_ob shape={obs0.shape}")


@torch.no_grad()
def evaluate(args, spec: EnvSpec):
    """Run the full plan-and-rollout eval loop for one environment and dump metrics."""
    import mujoco

    device = args.device
    eval_name = args.eval_name
    eval_dir = os.path.join("logs", spec.env_name, "eval", f"{eval_name}", str(args.seed), args.ev_cp_infer_t_type)
    mkdir(eval_dir)

    # load problems
    problems = load_eval_problems(spec.eval_probs_h5)
    start_states = problems["start_state"].astype(np.float32)
    goal_pos = problems["goal_pos"].astype(np.float32)

    is_antsoccer = spec.env_name.startswith("antsoccer")
    num_probs = start_states.shape[0]
    if args.ep_indices:
        ep_indices = [int(x) for x in str(args.ep_indices).split(",") if x.strip()]
    else:
        num_ep = num_probs if args.plan_n_ep < 0 else int(args.plan_n_ep)
        ep_indices = list(range(args.ep_st_idx, args.ep_st_idx + num_ep))
    if not ep_indices:
        raise ValueError("No eval episodes selected.")
    bad_indices = [idx for idx in ep_indices if idx < 0 or idx >= num_probs]
    if bad_indices:
        raise ValueError(f"Episode indices out of range for {num_probs} problems: {bad_indices}")

    # load normalizers
    train_normalizer = make_normalizer(spec.env_name, obs_select_dim=spec.plan_obs_select_dim)
    full_normalizer = make_normalizer(spec.env_name, obs_select_dim=None)
    
    # Identify env type for specific logic
    is_pointmaze = "pointmaze" in spec.env_name
    is_antsoccer = "antsoccer" in spec.env_name
    is_antmaze = "antmaze" in spec.env_name

    # load planner
    planner_dir = os.path.join("logs", spec.env_name, "planner", args.planner_name)
    planner_ckpt = checkpoint_path(planner_dir, args.planner_epoch)
    diffusion = build_planner_diffusion_for_env(spec).to(device)
    ckpt = torch.load(planner_ckpt, map_location=device, weights_only=False)
    if "ema" in ckpt:
        diffusion.load_state_dict(ckpt["ema"], strict=False)
    elif "planner_ema" in ckpt:
        diffusion.load_state_dict(ckpt["planner_ema"], strict=False)
    else:
        print("No EMA found in planner checkpoint", list(ckpt.keys()))
        exit(1)
    diffusion.eval()
    freeze_model(diffusion)

    diffusion.var_temp = float(args.var_temp)
    diffusion.condition_guidance_w = float(args.cond_w)
    diffusion.use_ddim = bool(args.use_ddim)
    diffusion.ddim_eta = float(args.ddim_eta)
    diffusion.ddim_num_inference_steps = int(args.ddim_steps)
    diffusion.ddim_t_power = float(args.ddim_t_power)

    ecd_markov_prior = None
    if str(args.ecd_markov_type).lower() in ["fitted", "fitted_gaussian", "gaussian"]:
        prior_path = args.ecd_prior_path
        if prior_path is None:
            prior_path = os.path.join("logs", spec.env_name, "ecd_prior", "gaussian_markov.pt")
        if not os.path.exists(prior_path):
            raise FileNotFoundError(
                f"ECD fitted Gaussian prior not found: {prior_path}. "
                f"Fit it first with: python -m src.compdiffuser.fit_ecd_prior --env {spec.env_name} --out {prior_path}"
            )
        ecd_markov_prior = FittedGaussianMarkovPrior.load(prior_path, map_location=device)
        if ecd_markov_prior.dim != diffusion.observation_dim:
            raise ValueError(
                f"ECD prior dimension mismatch: prior dim={ecd_markov_prior.dim}, "
                f"planner dim={diffusion.observation_dim}. Did you fit the prior for the same env/obs dims?"
            )
        print(
            f"[ECD] loaded fitted Gaussian Markov prior from {prior_path} "
            f"(dim={ecd_markov_prior.dim}, states={ecd_markov_prior.meta.n_states}, pairs={ecd_markov_prior.meta.n_pairs})"
        )

    # policy
    ecd_config = dict(
        base_scale=float(args.ecd_base_scale),
        base_dim_weights=args.ecd_base_dim_weights,
        react_scale=float(args.ecd_react_scale),
        react_clip=float(args.ecd_react_clip),
        react_dim_weights=args.ecd_react_dim_weights,
        markov_rho=float(args.ecd_markov_rho),
        markov_type=str(args.ecd_markov_type),
        markov_prior=ecd_markov_prior,
        prior_phi_weight=float(args.ecd_prior_phi_weight),
        prior_psi_weight=float(args.ecd_prior_psi_weight),
        prior_boundary_scale=float(args.ecd_prior_boundary_scale),
        prior_ridge=(None if args.ecd_prior_runtime_ridge is None else float(args.ecd_prior_runtime_ridge)),
        rank_type=str(args.ecd_rank_type),
        rank_t=args.ecd_rank_t,
        rank_dim_weights=args.ecd_rank_dim_weights,
        rank_invdyn_stride=int(args.ecd_rank_invdyn_stride),
        rank_invdyn_max_steps=int(args.ecd_rank_invdyn_max_steps),
        rank_invdyn_batch=int(args.ecd_rank_invdyn_batch),
        gsc_u=int(args.gsc_u),
        chunk_react_type=str(args.ecd_chunk_react_type),
        base_far_scale=args.ecd_base_far_scale,
        base_far_dist=args.ecd_base_far_dist,
    )
    rank_context = None
    policy = CompositionalPolicy(
        diffusion_model=diffusion,
        normalizer=train_normalizer,
        ev_n_comp=int(args.ev_n_comp),
        ev_top_n=int(args.ev_top_n),
        ev_pick_type=str(args.ev_pick_type),
        ev_cp_infer_t_type=str(args.ev_cp_infer_t_type),
        tj_blend_type=str(args.tjb_blend_type),
        tj_exp_beta=float(args.tjb_exp_beta),
        ecd_config=ecd_config,
        rank_context=rank_context,
    )

    # load invdyn
    inv_model = None
    if not is_pointmaze:
        if args.invdyn_name is None:
            raise ValueError("--invdyn_name is required for non-pointmaze environments.")
        inv_dir = os.path.join("logs", spec.env_name, "invdyn", args.invdyn_name)
        inv_ckpt = checkpoint_path(inv_dir, args.invdyn_epoch)
        tmp_env = ogb_make_env(spec.env_name)
        dset = ogb_load_train_dataset(tmp_env)
        act_dim = dset["actions"].shape[1]
        obs_dim = dset["observations"].shape[1]
        try:
            tmp_env.close()
        except Exception:
            pass

        inv_model = build_invdyn_model_for_env(spec, obs_dim=obs_dim, act_dim=act_dim).to(device)
        ckpt_inv = torch.load(inv_ckpt, map_location=device, weights_only=False)
        inv_model.load_state_dict(ckpt_inv["model"])
        if int(args.is_inv_train_mode) == 1:
            inv_model.train(True)
            for p in inv_model.parameters():
                p.requires_grad = False
        else:
            inv_model.eval()
            freeze_model(inv_model)

    # env for rollout
    env = ogb_make_env(spec.env_name)
    maze_repair_context = None
    if ("maze" in spec.env_name) and str(args.maze_repair_plan) == "shortest_path":
        maze_unwrapped = env.unwrapped
        maze_repair_context = {
            "maze_map": np.asarray(maze_unwrapped.maze_map).copy(),
            "maze_unit": ogb_maze_unit(maze_unwrapped),
            "offset_x": ogb_offset_x(maze_unwrapped),
            "offset_y": ogb_offset_y(maze_unwrapped),
        }

    # collect for metrics + visualization
    ep_is_suc: List[bool] = []
    ep_cnt_repl: List[int] = []
    ep_steps: List[int] = []
    ep_plan_xy: List[np.ndarray] = []
    ep_roll_xy: List[np.ndarray] = []
    ep_roll_obs_sel: List[np.ndarray] = []
    ep_st_xy: List[np.ndarray] = []
    ep_gl_xy: List[np.ndarray] = []
    ep_tracking_dist: List[np.ndarray] = []
    ep_goal_dist: List[float] = []
    ep_task_goal_dist: List[float] = []

    goal_sel_idxs = tuple(range(spec.goal_dim))
    obs_sel = spec.plan_obs_select_dim
    if args.repl_used_idxs:
        replan_used_idxs = tuple(int(x) for x in str(args.repl_used_idxs).split(",") if x.strip())
        bad_repl_idxs = [i for i in replan_used_idxs if i < 0 or i >= len(obs_sel)]
        if bad_repl_idxs:
            raise ValueError(
                f"--repl_used_idxs are indices into planner observation length {len(obs_sel)}; "
                f"out-of-range: {bad_repl_idxs}."
            )
    else:
        replan_used_idxs = tuple(i for i, d in enumerate(obs_sel) if int(d) in (0, 1))
    if not replan_used_idxs:
        replan_used_idxs = tuple(range(min(2, len(obs_sel))))
    if inv_model is not None:
        policy.rank_context.update(
            {
                "inv_model": inv_model,
                "full_normalizer": full_normalizer,
                "obs_select_dim": tuple(obs_sel),
            }
        )

    # For replanning schedule (m_2 from your repo)
    repl_thres = float(args.repl_thres if args.repl_thres is not None else spec.eval_repl_thres)
    repl_max_n = int(args.repl_max_n if args.repl_max_n is not None else spec.eval_max_n_repl)
    ada_minus_n_wp = int(args.ada_dist_minus_n_wp if args.ada_dist_minus_n_wp is not None else spec.eval_ada_minus_n_wp)
    cond2_extra = int(args.cond2_extra if args.cond2_extra is not None else spec.eval_cond2_extra)
    n_max_steps = int(args.n_max_steps if args.n_max_steps is not None else spec.eval_n_max_steps)

    fused_alloc = int(args.fused_traj_alloc)

    for ep_i in ep_indices:
        st_state = start_states[ep_i]
        gl_state = goal_pos[ep_i]

        env.reset()
        set_start_state(env, st_state)
        if is_antsoccer:
            env.set_goal(goal_xy=gl_state[15:17])
        else:
            env.set_goal(goal_xy=gl_state[:2])
        mujoco.mj_forward(env.model, env.data)

        obs_cur = env.get_ob().copy().astype(np.float32)
        st_xy = obs_cur[list(obs_sel)].copy()
        gl_xy = gl_state[list(obs_sel)].copy()

        fused = np.zeros((fused_alloc, len(obs_sel)), dtype=np.float32)
        all_plans: List[np.ndarray] = []
        cnt_repl = 0
        prev_dfu_wp_idx = 0
        prev_n_comp = int(args.ev_n_comp)

        is_suc = False
        extra_after_suc = 0
        rollout_xy = [obs_cur[:2].copy()]
        rollout_obs_sel = [obs_cur[list(obs_sel)].copy()]

        # initial plan at t=0
        policy.n_comp = prev_n_comp
        policy.rank_context["episode_start_plan_obs"] = st_xy.copy()
        policy.rank_context["episode_goal_plan_obs"] = gl_xy.copy()
        policy.rank_context["current_full_obs"] = obs_cur.copy()
        pick_traj = policy.plan(st_xy, gl_xy, b_s=int(args.b_size_per_prob))
        if maze_repair_context is not None:
            pick_traj = shortest_path_plan_xy(st_xy, gl_xy, maze_repair_context, n_points=len(pick_traj))
        fused[:len(pick_traj)] = pick_traj
        fused[len(pick_traj):] = pick_traj[-1]
        all_plans.append(pick_traj)
        last_action_goal_cur = fused[0].astype(np.float32).copy()

        for i_et in range(n_max_steps):
            wp_idx = i_et // int(args.n_act_per_waypnt)
            is_wp_boundary = (wp_idx * int(args.n_act_per_waypnt)) == i_et

            do_plan = False
            if args.is_replan and args.is_replan != "none" and is_wp_boundary and len(all_plans) > 0 and i_et > 0:
                if str(args.repl_dist_ref) == "previous":
                    goal_cur = last_action_goal_cur
                else:
                    goal_cur = fused[min(wp_idx, fused_alloc - 1)]
                obs_plan_cur = obs_cur[list(obs_sel)]
                dist = np.linalg.norm(obs_plan_cur[list(replan_used_idxs)] - goal_cur[list(replan_used_idxs)])
                cond1 = dist > repl_thres
                cond2 = (wp_idx - prev_dfu_wp_idx - cond2_extra) > len(all_plans[-1])
                do_plan = (cond1 or cond2) and (cnt_repl < repl_max_n)

            if do_plan:
                cnt_repl += 1
                tmp_cnt_wp = wp_idx - prev_dfu_wp_idx
                tmp_v1 = max(0, tmp_cnt_wp - ada_minus_n_wp)
                prev_hzn = len(all_plans[-1])
                tmp_n_comp = math.ceil((1.0 - tmp_v1 / max(1, prev_hzn)) * prev_n_comp)
                tmp_n_comp = max(1, tmp_n_comp)
                prev_n_comp = tmp_n_comp
                policy.n_comp = tmp_n_comp

                st_xy = obs_cur[list(obs_sel)].copy()
                policy.rank_context["current_full_obs"] = obs_cur.copy()
                pick_traj = policy.plan(st_xy, gl_xy, b_s=int(args.b_size_per_prob))
                if maze_repair_context is not None:
                    pick_traj = shortest_path_plan_xy(st_xy, gl_xy, maze_repair_context, n_points=len(pick_traj))
                end_idx = wp_idx + len(pick_traj)
                if end_idx > fused_alloc:
                    pick_traj = pick_traj[: (fused_alloc - wp_idx)]
                    end_idx = fused_alloc
                fused[wp_idx:end_idx] = pick_traj
                fused[end_idx:] = pick_traj[-1]
                all_plans.append(pick_traj)
                prev_dfu_wp_idx = wp_idx

            # invdyn action toward waypoint
            wp_idx_safe = min(wp_idx + int(args.action_wp_offset), fused_alloc - 1)
            goal_cur = fused[wp_idx_safe].astype(np.float32)
            last_action_goal_cur = goal_cur.copy()

            if is_pointmaze:
                act_nm = (goal_cur - obs_cur[list(obs_sel)]) * float(args.pd_gain)
                act = full_normalizer.unnormalize(act_nm, "actions").astype(np.float32)
            else:
                obs_nm = full_normalizer.normalize(obs_cur, "observations")
                goal_nm = train_normalizer.normalize(goal_cur, "observations")

                obs_t = torch.from_numpy(obs_nm).to(device=device, dtype=torch.float32)[None, :]
                goal_t = torch.from_numpy(goal_nm).to(device=device, dtype=torch.float32)[None, :]
                act_nm = inv_model(obs_t, goal_t).detach().cpu().numpy()[0]
                act = full_normalizer.unnormalize(act_nm, "actions").astype(np.float32)

            obs_next, rew, terminated, truncated, info = env.step(act)
            obs_cur = obs_next.astype(np.float32)
            rollout_xy.append(obs_cur[:2].copy())
            rollout_obs_sel.append(obs_cur[list(obs_sel)].copy())

            is_suc = bool(info.get("success", False)) or is_suc
            if is_suc:
                extra_after_suc += 1
                if extra_after_suc >= 30:
                    break

        ep_is_suc.append(bool(is_suc))
        ep_cnt_repl.append(int(cnt_repl))
        ep_steps.append(len(rollout_xy))
        rollout_arr = np.asarray(rollout_xy, dtype=np.float32)
        rollout_obs_sel_arr = np.asarray(rollout_obs_sel, dtype=np.float32)
        track_len = min(len(rollout_obs_sel_arr), len(fused))
        plan_for_rollout = fused[:track_len, :].copy()
        track_dist = np.linalg.norm(
            rollout_obs_sel_arr[:track_len, :][:, list(replan_used_idxs)] - plan_for_rollout[:, list(replan_used_idxs)],
            axis=1,
        )
        ep_plan_xy.append(plan_for_rollout)  # truncate to rollout length for plotting
        ep_roll_xy.append(rollout_arr)
        ep_roll_obs_sel.append(rollout_obs_sel_arr)
        ep_st_xy.append(st_state[:2].copy())
        ep_gl_xy.append(gl_state[:2].copy())
        ep_tracking_dist.append(track_dist.astype(np.float32))
        ep_goal_dist.append(float(np.linalg.norm(obs_cur[list(obs_sel)][list(replan_used_idxs)] - gl_xy[list(replan_used_idxs)])))
        if is_antsoccer:
            ep_task_goal_dist.append(float(np.linalg.norm(obs_cur[15:17] - gl_state[15:17])))
        else:
            ep_task_goal_dist.append(float(ep_goal_dist[-1]))

        print(f"[eval] ep={ep_i} success={is_suc} repl={cnt_repl} steps={len(rollout_xy)}")

    # ---- metrics ----
    sr = float(np.mean(ep_is_suc) * 100.0)
    print(f"\n[eval] Success rate: {sr:.1f}% ({np.sum(ep_is_suc)}/{len(ep_is_suc)})")

    # SR by replan-count bound
    sr_by_bound = {}
    for k in range(0, repl_max_n + 1):
        mask = np.array(ep_cnt_repl) <= k
        if mask.sum() == 0:
            sr_k = None
        else:
            sr_k = float(np.mean(np.array(ep_is_suc)[mask]) * 100.0)
        sr_by_bound[k] = {"n": int(mask.sum()), "sr": sr_k}
    print("[eval] Success rate by repl-count bound (<=k):")
    for k, v in sr_by_bound.items():
        print(f"  k={k:2d}  n={v['n']:3d}  sr={v['sr']}")

    # sampling time summary (max n_comp bucket)
    ncp_times = np.array(policy.ncp_pred_time_list, dtype=np.float32) if policy.ncp_pred_time_list else None
    sampling_time_summary = None
    if ncp_times is not None and len(ncp_times) > 0:
        max_ncp = int(np.max(ncp_times[:, 0]))
        times = ncp_times[ncp_times[:, 0] == max_ncp, 1]
        if len(times) > 2:
            times = times[2:]
        sampling_time_summary = {"n_comp": max_ncp, "n": int(len(times)), "mean_s": float(times.mean()), "std_s": float(times.std())}
        print(f"[eval] Sampling time (n_comp={max_ncp}): mean={times.mean():.4f}s std={times.std():.4f}s n={len(times)}")

    all_tracking = np.concatenate(ep_tracking_dist) if ep_tracking_dist else np.asarray([], dtype=np.float32)
    tracking_summary = None
    if len(all_tracking) > 0:
        tracking_summary = {
            "mean": float(np.mean(all_tracking)),
            "p50": float(np.percentile(all_tracking, 50)),
            "p90": float(np.percentile(all_tracking, 90)),
            "p99": float(np.percentile(all_tracking, 99)),
            "max": float(np.max(all_tracking)),
            "final_goal_dist_mean": float(np.mean(ep_goal_dist)),
            "final_goal_dist_p50": float(np.percentile(ep_goal_dist, 50)),
            "final_task_goal_dist_mean": float(np.mean(ep_task_goal_dist)),
            "final_task_goal_dist_p50": float(np.percentile(ep_task_goal_dist, 50)),
        }
        print(f"[eval] Plan tracking summary: {tracking_summary}")

    ecd_debug_summary = None
    if policy.ecd_step_stats_list:
        base_norms = []
        reaction_norms = []
        for plan_stats in policy.ecd_step_stats_list:
            for step_stats in plan_stats:
                base_norms.append(float(step_stats.get("base_norm", 0.0)))
                reaction_norms.append(float(step_stats.get("reaction_norm", 0.0)))
        base_arr = np.asarray(base_norms, dtype=np.float64)
        react_arr = np.asarray(reaction_norms, dtype=np.float64)
        ratios = react_arr / np.maximum(base_arr, 1e-12)
        ecd_debug_summary = {
            "num_plans": int(len(policy.ecd_step_stats_list)),
            "num_steps": int(len(base_norms)),
            "reaction_base_ratio_mean": float(ratios.mean()),
            "reaction_base_ratio_p50": float(np.percentile(ratios, 50)),
            "reaction_base_ratio_p90": float(np.percentile(ratios, 90)),
            "base_norm_mean": float(base_arr.mean()),
            "reaction_norm_mean": float(react_arr.mean()),
        }
        if policy.ecd_rank_score_summary_list:
            rank_means = np.asarray([x["mean"] for x in policy.ecd_rank_score_summary_list], dtype=np.float64)
            rank_spreads = np.asarray(
                [x["max"] - x["min"] for x in policy.ecd_rank_score_summary_list], dtype=np.float64
            )
            ecd_debug_summary.update(
                {
                    "rank_score_mean": float(rank_means.mean()),
                    "rank_score_spread_mean": float(rank_spreads.mean()),
                }
            )
        print(f"[eval] ECD debug summary: {ecd_debug_summary}")

    # ---- visualization ----
    samples_path = os.path.join(eval_dir, f"samples_{args.ev_cp_infer_t_type}.png")
    save_eval_samples_plot(samples_path, env, ep_st_xy, ep_gl_xy, ep_plan_xy, ep_roll_xy)
    tracking_path = os.path.join(eval_dir, f"tracking_{args.ev_cp_infer_t_type}.png")
    save_eval_tracking_plot(tracking_path, env, ep_st_xy, ep_gl_xy, ep_plan_xy, ep_roll_xy, ep_tracking_dist)

    np.savez_compressed(
        os.path.join(eval_dir, "trajectories.npz"),
        ep_indices=np.asarray(ep_indices, dtype=np.int32),
        success=np.asarray(ep_is_suc, dtype=np.bool_),
        replans=np.asarray(ep_cnt_repl, dtype=np.int32),
        steps=np.asarray(ep_steps, dtype=np.int32),
        plan_xy=np.asarray(ep_plan_xy, dtype=object),
        rollout_xy=np.asarray(ep_roll_xy, dtype=object),
        rollout_obs_select=np.asarray(ep_roll_obs_sel, dtype=object),
        tracking_dist=np.asarray(ep_tracking_dist, dtype=object),
        task_goal_dist=np.asarray(ep_task_goal_dist, dtype=np.float32),
        start_xy=np.asarray(ep_st_xy, dtype=np.float32),
        goal_xy=np.asarray(ep_gl_xy, dtype=np.float32),
    )

    # ---- dump results.json ----
    results = {
        "env_name": spec.env_name,
        "env_spec": asdict(spec),
        "eval_args": vars(args),
        "summary": {
            "num_ep": int(len(ep_is_suc)),
            "success_rate": sr,
            "avg_replans": float(np.mean(ep_cnt_repl)),
            "avg_steps": float(np.mean(ep_steps)),
            "sr_by_replan_bound": sr_by_bound,
            "sampling_time_summary": sampling_time_summary,
            "tracking_summary": tracking_summary,
            "ecd_debug_summary": ecd_debug_summary,
        },
        "per_episode": [
            {
                "ep_idx": int(ep_indices[i]),
                "success": bool(ep_is_suc[i]),
                "replans": int(ep_cnt_repl[i]),
                "steps": int(ep_steps[i]),
                "track_mean": float(np.mean(ep_tracking_dist[i])),
                "track_p90": float(np.percentile(ep_tracking_dist[i], 90)),
                "track_max": float(np.max(ep_tracking_dist[i])),
                "final_goal_dist": float(ep_goal_dist[i]),
                "final_task_goal_dist": float(ep_task_goal_dist[i]),
            }
            for i in range(len(ep_is_suc))
        ],
    }
    save_json(results, os.path.join(eval_dir, "results.json"))

    try:
        env.close()
    except Exception:
        pass


def main():
    """CLI entry point for evaluation."""
    p = argparse.ArgumentParser()
    p.add_argument("--spec_csv", type=str, default="ogb_env_spec.csv")
    p.add_argument("--env", type=str, default="antmaze-giant-stitch-v0", help="Environment name, must be a key in ogb_env_spec.csv")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_name", type=str, default="compdiffuser")
    p.add_argument("--planner_name", type=str, required=True)
    p.add_argument("--planner_epoch", type=str, default="latest")
    p.add_argument("--invdyn_name", type=str, default=None)
    p.add_argument("--invdyn_epoch", type=str, default="latest")
    p.add_argument("--plan_n_ep", type=int, default=20)
    p.add_argument("--ep_st_idx", type=int, default=0)
    p.add_argument("--ep_indices", type=str, default=None,
                   help="Optional comma-separated eval problem indices. Overrides --plan_n_ep/--ep_st_idx.")
    p.add_argument("--seed", type=int, default=0)

    # diffusion sampling
    p.add_argument("--ev_n_comp", type=int, default=None, help="Default from csv if not set.")
    p.add_argument("--var_temp", type=float, default=1.0)
    p.add_argument("--cond_w", type=float, default=2.0)
    p.add_argument("--use_ddim", type=int, default=1)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=1.0)
    p.add_argument("--ddim_t_power", type=float, default=1.0,
                   help="Shared DDIM timestep power grid; 1.0 matches CD uniform grid.")

    # execution
    p.add_argument("--n_act_per_waypnt", type=int, default=1)
    p.add_argument("--action_wp_offset", type=int, default=0,
                   help="Optional lookahead offset into the fused plan when selecting the invdyn/PD action subgoal.")
    p.add_argument("--b_size_per_prob", type=int, default=40)
    p.add_argument("--pd_gain", type=float, default=5.0, help="PointMaze-only PD gain toward waypoints.")

    p.add_argument("--is_replan", type=str, default="ada_dist", choices=["ada_dist", "none"])
    p.add_argument("--n_max_steps", type=int, default=None)
    p.add_argument("--repl_max_n", type=int, default=None,
                   help="Max replans. Default None uses ogb_env_spec.csv, matching the per-env CD setting.")
    p.add_argument("--repl_thres", type=float, default=None)
    p.add_argument("--repl_used_idxs", type=str, default=None,
                   help="Comma-separated planner-observation indices for ada_dist tracking; e.g. 0,1,15,16 for AntSoccer ant+ball xy.")
    p.add_argument("--repl_dist_ref", type=str, default="upcoming", choices=["upcoming", "previous"],
                   help="Waypoint reference used for ada_dist. upcoming preserves current eval behavior; previous matches the original planner's just-attempted waypoint check.")
    p.add_argument("--ada_dist_minus_n_wp", type=int, default=None)
    p.add_argument("--cond2_extra", type=int, default=None)
    p.add_argument("--maze_repair_plan", type=str, default="none", choices=["none", "shortest_path"],
                   help="Pointmaze-only postprocess; shortest_path replaces sampled plans with maze BFS waypoints.")
    p.add_argument("--fused_traj_alloc", type=int, default=10000)

    p.add_argument("--ev_top_n", type=int, default=5)
    p.add_argument("--ev_pick_type", type=str, default="first", choices=["first", "rand"])
    p.add_argument("--ev_cp_infer_t_type", type=str, default="interleave",
                   choices=["interleave", "ecd_chunk", "gsc_resampling"],
                   help="interleave = CompDiffuser baseline; ecd_chunk = ECD; gsc_resampling = CDGS baseline.")
    p.add_argument("--gsc_u", type=int, default=4,
                   help="CDGS (gsc_resampling) inner resampling rounds U per reverse step (~U x compute). Paper uses 4.")
    p.add_argument("--tjb_blend_type", type=str, default="exp")
    p.add_argument("--tjb_exp_beta", type=float, default=2.0)

    p.add_argument("--is_inv_train_mode", type=int, default=0,
               help="If 1, keep invdyn model in train() mode (matches original AntSoccer eval).")

    # ECD-specific controls. These do not change the number of local denoiser
    # calls; they only control the added reaction channel.
    p.add_argument("--ecd_base_scale", type=float, default=1.0)
    p.add_argument("--ecd_base_far_scale", type=float, default=None,
                   help="Optional AntSoccer distance-conditioned base scale used when ball-goal distance exceeds --ecd_base_far_dist.")
    p.add_argument("--ecd_base_far_dist", type=float, default=None,
                   help="Ball-goal distance threshold for --ecd_base_far_scale; disabled when unset.")
    p.add_argument("--ecd_base_dim_weights", type=str, default=None,
                   help="Optional per-dimension weights applied only to the ECD bridge/base update, e.g. 'xy=1,rest=0'.")
    p.add_argument("--ecd_react_scale", type=float, default=1.0)
    p.add_argument("--ecd_react_clip", type=float, default=0.0,
                   help="Coordinatewise clipping for the reaction message; <=0 disables.")
    p.add_argument("--ecd_react_dim_weights", type=str, default=None,
                   help="Optional per-dimension weights applied only to ECD reaction/JVP messages, e.g. 'xy=1,rest=0.1'.")
    p.add_argument("--ecd_markov_rho", type=float, default=0.25,
                   help="Boundary coupling strength in the block-tridiagonal Markov approximation.")
    p.add_argument("--ecd_markov_type", type=str, default="laplacian",
                   choices=["laplacian", "fitted_gaussian"],
                   help="Markov surrogate for the ecd_chunk reaction. laplacian needs no fit; fitted_gaussian loads phi/psi statistics.")
    p.add_argument("--ecd_prior_path", type=str, default=None,
                   help="Path to gaussian_markov.pt. Defaults to logs/<env>/ecd_prior/gaussian_markov.pt for fitted_gaussian.")
    p.add_argument("--ecd_prior_phi_weight", type=float, default=1.0,
                   help="Unary fitted-state precision weight in the fitted Gaussian Markov solve.")
    p.add_argument("--ecd_prior_psi_weight", type=float, default=0.25,
                   help="Adjacent-pair fitted precision weight in the fitted Gaussian Markov solve and boundary G block.")
    p.add_argument("--ecd_prior_boundary_scale", type=float, default=1.0,
                   help="Extra scale on the fitted Gaussian boundary reaction message -G^T u.")
    p.add_argument("--ecd_prior_runtime_ridge", type=float, default=None,
                   help="Optional extra runtime ridge for noised covariance inversion; None uses the fitted prior ridge.")
    p.add_argument("--ecd_rank_type", type=str, default="overlap",
                   choices=["overlap", "overlap_invdyn"],
                   help="Candidate ranking. overlap = overlap-region mismatch; overlap_invdyn adds invdyn action demand.")
    p.add_argument("--ecd_rank_t", type=int, default=None,
                   help="Unused legacy ranking timestep flag; kept for script compatibility.")
    p.add_argument("--ecd_rank_dim_weights", type=str, default=None,
                   help="Optional overlap-rank dimension weights. Examples: 'xy=1,rest=0.1', '1,0.1', or 15 comma floats.")
    p.add_argument("--ecd_rank_invdyn_stride", type=int, default=4,
                   help="Stride over blended plan waypoints for invdyn-aware ECD ranking.")
    p.add_argument("--ecd_rank_invdyn_max_steps", type=int, default=96,
                   help="Maximum strided waypoint transitions scored by invdyn-aware ranking; <=0 scores all.")
    p.add_argument("--ecd_rank_invdyn_batch", type=int, default=4096,
                   help="Batch size for invdyn-aware ECD ranking.")
    p.add_argument("--ecd_chunk_react_type", type=str, default="markov",
                   choices=["markov", "none"],
                   help="Reaction term inside ecd_chunk: markov approximation, or none.")


    args = p.parse_args()

    set_seed(args.seed)
    specs = load_env_specs(args.spec_csv)
    if args.env not in specs:
        raise ValueError(f"--env {args.env} not found in {args.spec_csv}. Available: {list(specs.keys())}")
    spec = specs[args.env]
    print(f"[main] loaded env spec for {args.env}: {spec}")

    if args.ev_n_comp is None:
        args.ev_n_comp = spec.eval_default_n_comp

    evaluate(args, spec)


if __name__ == "__main__":
    main()
