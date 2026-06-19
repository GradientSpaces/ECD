"""
Ported CompDiffuser (Stgl/Sml) training + inference for OGBench locomaze tasks,
extended to multiple environments via ogb_env_spec.csv (read at runtime).

Log layout:
    logs/<env_name>/planner/<planner_name>/{args.json, state_x.pt}
    logs/<env_name>/planner/<planner_name>/val/{step_x.png}
    logs/<env_name>/invdyn/<invdyn_name>/{args.json, state_x.pt}
    logs/<env_name>/eval/<eval_name>_<timestamp>/{results.json, samples_x.png}
"""

from __future__ import annotations

import os
import time
import argparse
import random
from dataclasses import asdict

import numpy as np
import torch
import wandb

from .dataset.common import EnvSpec, DatasetNormalizer, load_env_specs
from .dataset.ogbench import OgBPlanningDataset, ogb_make_env, make_normalizer
from .utils.ema import EMA
from .utils.visualization import save_training_validation_plot

from .planner import ChunkDiffusion, build_planner_diffusion_for_env
from .policy import CompositionalPolicy
from .eval import load_eval_problems
from .utils.io import (
    mkdir,
    save_json,
    load_text,
    env_name_short,
    find_latest_checkpoint,
    get_lr,
    save_text,
    to_device,
)

# Torch perf defaults
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def train_planner(args, spec: EnvSpec):
    """Train the CompDiffuser planner for one environment, with checkpointing and optional W&B/validation."""
    device = args.device
    planner_name = args.planner_name
    planner_dir = os.path.join("logs", spec.env_name, "planner", planner_name)
    mkdir(planner_dir)
    save_json(vars(args) | {"env_spec": asdict(spec)}, os.path.join(planner_dir, "args.json"))

    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume_path = find_latest_checkpoint(planner_dir)
        if resume_path is not None:
            print(f"[planner] auto-resume from {resume_path}")

    wandb_id_path = os.path.join(planner_dir, "wandb_run_id.txt")
    wandb_resume_id = load_text(wandb_id_path) if resume_path is not None else None

    wandb_run = None
    train_normalizer = make_normalizer(spec.env_name, obs_select_dim=spec.plan_obs_select_dim)
    dataset = OgBPlanningDataset(spec, train_normalizer)

    diffusion = build_planner_diffusion_for_env(spec).to(device)
    ema = EMA(args.ema_decay)
    ema_model = build_planner_diffusion_for_env(spec).to(device)
    ema_model.load_state_dict(diffusion.state_dict())

    opt = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    scheduler = None  # could add lr scheduler here if desired

    step = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        diffusion.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "scheduler" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "rng_state" in ckpt:
            random.setstate(ckpt["rng_state"])
        if "np_rng_state" in ckpt:
            np.random.set_state(ckpt["np_rng_state"])
        # Torch RNG state is intentionally not restored on resume.
        step = int(ckpt["step"])
        if wandb_resume_id is None:
            wandb_resume_id = ckpt.get("wandb_id")
        print(f"[planner] resumed from {resume_path} (step={step})")

    if args.wandb:
        if wandb is None:
            print("[wandb] wandb not installed; skipping logging.")
        else:
            run_name = f"{env_name_short(spec.env_name)}_{planner_name}"
            wandb_init_kwargs = dict(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                dir=planner_dir,
                config=vars(args) | {"env_spec": asdict(spec)},
                mode=args.wandb_mode,
            )
            if wandb_resume_id:
                wandb_init_kwargs["id"] = wandb_resume_id
                wandb_init_kwargs["resume"] = "allow"
            wandb_run = wandb.init(**wandb_init_kwargs)
            if wandb_run is not None:
                save_text(wandb_id_path, wandb_run.id)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )

    label_freq = max(1, int(args.n_train_steps // args.n_saves))
    t_start = time.time()

    while step < args.n_train_steps:
        for batch in loader:
            if step >= args.n_train_steps:
                break

            diffusion.train()

            for _ in range(args.grad_accum):
                obs_trajs, _act_trajs, cond = batch
                obs_trajs = to_device(obs_trajs, device)
                cond = to_device(cond, device)
                loss, _ = diffusion.loss(obs_trajs, cond)
                (loss / args.grad_accum).backward()

            opt.step()
            opt.zero_grad(set_to_none=True)

            if step < args.step_start_ema:
                ema_model.load_state_dict(diffusion.state_dict())
            elif (step % args.update_ema_every) == 0:
                ema.update_model_average(ema_model, diffusion)

            if (step % args.save_freq) == 0:
                label = (step // label_freq) * label_freq
                save_path = os.path.join(planner_dir, f"state_{label}.pt")
                # remove the previous saved checkpoint to save space
                prev_label = label - label_freq
                if prev_label >= 0:
                    prev_path = os.path.join(planner_dir, f"state_{prev_label}.pt")
                    if os.path.exists(prev_path):
                        os.remove(prev_path)
                        print(f"[planner] removed previous checkpoint {prev_path}")
                ckpt = {
                    "step": step,
                    "model": diffusion.state_dict(),
                    "ema": ema_model.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "rng_state": random.getstate(),
                    "np_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "torch_cuda_rng_state": torch.cuda.random.get_rng_state_all() if torch.cuda.is_available() else None,
                    "wandb_id": wandb_run.id if wandb_run is not None else None,
                }
                torch.save(ckpt, save_path)
                print(f"[planner] saved new checkpoint {save_path}")

            if (step % args.log_freq) == 0:
                elapsed_hours = (time.time() - t_start) / 3600
                print(f"[planner] step={step} loss={loss.item():.6f} lr={get_lr(opt):.2e} elapsed={elapsed_hours:.2f}h")
                if wandb_run is not None:
                    wandb.log({"train/loss": float(loss.item()), "train/lr": get_lr(opt), "elapsed_hours": elapsed_hours}, step=step)

            if (args.val_freq > 0 and (step % args.val_freq) == 0):
                try:
                    print(f"[planner][val] step={step} ...")
                    val_path = validate(spec, planner_name, ema_model, train_normalizer, step, device)
                    if wandb_run is not None:
                        wandb.log({"val/vis": wandb.Image(val_path)}, step=step)
                except Exception as e:
                    print(f"[planner][val] failed: {e}")

            step += 1
    if wandb_run is not None:
        wandb_run.finish()


def validate(
    spec: EnvSpec,
    planner_name: str,
    diffusion_ema: ChunkDiffusion,
    train_normalizer: DatasetNormalizer,
    step: int,
    device: str,
    b_s: int = 40,
):
    """
    Training-time validation visualization:
      - no replan, no env interaction
      - first 3 problems from eval hdf5
      - produce interleave plans, 3 subplots row
    """
    problems = load_eval_problems(spec.eval_probs_h5)
    st_full = problems["start_state"][[0, 30, 60]].astype(np.float32)
    gl_full = problems["goal_pos"][[0, 30, 60]].astype(np.float32)

    env = ogb_make_env(spec.env_name)
    starts_xy = st_full[:, list(spec.plan_obs_select_dim)]
    goals_xy = gl_full[:, list(spec.plan_obs_select_dim)]

    # interleave plan
    diffusion_ema.eval()
    diffusion_ema.var_temp = 1.0
    diffusion_ema.condition_guidance_w = 2.0
    diffusion_ema.use_ddim = True
    diffusion_ema.ddim_eta = 1.0
    diffusion_ema.ddim_num_inference_steps = 50

    pol_interleave = CompositionalPolicy(
        diffusion_model=diffusion_ema,
        normalizer=train_normalizer,
        ev_n_comp=spec.eval_default_n_comp,
        ev_top_n=5,
        ev_pick_type="first",
        ev_cp_infer_t_type="interleave",
        tj_blend_type="exp",
        tj_exp_beta=2.0,
    )

    plans_interleave = []
    for i in range(3):
        plans_interleave.append(pol_interleave.plan(starts_xy[i], goals_xy[i], b_s=b_s))

    save_path = os.path.join("logs", spec.env_name, "planner", planner_name, "val", f"step_{step}.png")
    save_training_validation_plot(save_path, env, starts_xy, goals_xy, plans_interleave)

    try:
        env.close()
    except Exception:
        pass
    return save_path


def main():
    """CLI entry point for planner training."""
    p = argparse.ArgumentParser()
    p.add_argument("--spec_csv", type=str, default="ogb_env_spec.csv")
    p.add_argument("--env", type=str, default="antmaze-giant-stitch-v0", help="Environment name, must be a key in ogb_env_spec.csv")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--planner_name", type=str, default="baseline")

    p.add_argument("--batch_size", type=int, default=128, help="Batch size")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--step_start_ema", type=int, default=4000)
    p.add_argument("--update_ema_every", type=int, default=10)
    p.add_argument("--n_train_steps", type=int, default=2_000_000)

    p.add_argument("--save_freq", type=int, default=5000)
    p.add_argument("--n_saves", type=int, default=5)
    p.add_argument("--log_freq", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--auto_resume", type=int, default=1, help="Auto-resume from latest checkpoint if found (1/0).")
    p.add_argument("--val_freq", type=int, default=5000, help="If >0, save val plot every val_freq steps (uses eval hdf5 first 3 problems).")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="compdiffuser")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    args = p.parse_args()

    specs = load_env_specs(args.spec_csv)
    if args.env not in specs:
        raise ValueError(f"--env {args.env} not found in {args.spec_csv}. Available: {list(specs.keys())}")
    spec = specs[args.env]
    print(f"[train] loaded env spec for {args.env}: {spec}")

    train_planner(args, spec)


if __name__ == "__main__":
    main()
