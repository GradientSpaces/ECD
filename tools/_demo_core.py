"""Shared helpers for the ECD demo (used by demo.ipynb). PointMaze: planner predicts x-y directly."""
import os
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np, torch
from ecd.dataset.common import load_env_specs
from ecd.dataset.ogbench import make_normalizer
from ecd.planner import build_planner_diffusion_for_env
from ecd.policy import CompositionalPolicy
from ecd.utils.io import checkpoint_path, freeze_model
from ecd.eval import load_eval_problems

# Full ECD config with sensible defaults; override only what differs between CD and ECD.
DEFAULT_ECD = dict(
    base_scale=1.0, base_dim_weights=None, react_scale=1.0, react_clip=0.0, react_dim_weights=None,
    markov_rho=0.25, markov_type="laplacian", markov_prior=None, prior_phi_weight=1.0,
    prior_psi_weight=0.25, prior_boundary_scale=1.0, prior_ridge=None, rank_type="overlap",
    rank_t=None, rank_dim_weights=None, rank_invdyn_stride=4, rank_invdyn_max_steps=96,
    rank_invdyn_batch=4096, chunk_react_type="markov",
    base_far_scale=None, base_far_dist=None,
)

def load_planner(env, device="cuda", spec_csv="ogb_env_spec.csv"):
    spec = load_env_specs(spec_csv)[env]
    normalizer = make_normalizer(env, obs_select_dim=spec.plan_obs_select_dim)
    diffusion = build_planner_diffusion_for_env(spec).to(device)
    ckpt = torch.load(checkpoint_path(f"logs/{env}/planner/planner", "latest"),
                      map_location=device, weights_only=False)
    diffusion.load_state_dict(ckpt.get("ema", ckpt.get("planner_ema")), strict=False)
    diffusion.eval(); freeze_model(diffusion)
    diffusion.var_temp, diffusion.condition_guidance_w = 1.0, 2.0
    diffusion.use_ddim, diffusion.ddim_eta = True, 1.0
    diffusion.ddim_num_inference_steps, diffusion.ddim_t_power = 50, 1.0
    return spec, normalizer, diffusion

def make_policy(diffusion, normalizer, n_comp, infer_type, **ecd_overrides):
    """infer_type='interleave' -> CD baseline; 'ecd_chunk' -> ECD. ecd_overrides patch DEFAULT_ECD."""
    cfg = {**DEFAULT_ECD, **ecd_overrides}
    pol = CompositionalPolicy(
        diffusion_model=diffusion, normalizer=normalizer, ev_n_comp=n_comp, ev_top_n=5,
        ev_pick_type="first", ev_cp_infer_t_type=infer_type, tj_blend_type="exp",
        tj_exp_beta=2.0, ecd_config=cfg, rank_context=None,
    )
    return pol

@torch.no_grad()
def plan_xy(policy, start_xy, goal_xy, b_size=40):
    """Return a stitched (T, 2) x-y plan from start to goal."""
    policy.rank_context["episode_start_plan_obs"] = np.asarray(start_xy, np.float32).copy()
    policy.rank_context["episode_goal_plan_obs"] = np.asarray(goal_xy, np.float32).copy()
    policy.rank_context["current_full_obs"] = np.asarray(start_xy, np.float32).copy()
    return policy.plan(np.asarray(start_xy, np.float32), np.asarray(goal_xy, np.float32), b_s=b_size)
