"""Train and build the inverse-dynamics model used to convert plans into actions."""

import os
import time
import argparse
from dataclasses import asdict

import matplotlib.pyplot as plt
import torch
from torch import nn

from .dataset.common import EnvSpec, load_env_specs
from .dataset.ogbench import OgBInvDynDataset, make_normalizer, ogb_make_env, ogb_xy_to_ij
from .utils.ema import EMA
from .utils.io import save_json, mkdir, get_lr, to_device
from .model.mlp import InvDynMLP


def build_invdyn_model_for_env(spec: EnvSpec, obs_dim: int, act_dim: int, goal_dim: int = None) -> nn.Module:
    """Construct an inverse-dynamics MLP sized for the given environment family."""
    goal_dim = spec.goal_dim if goal_dim is None else int(goal_dim)
    input_dim = obs_dim + goal_dim

    if spec.family in ["antM", "pointM"]:
        inv_m_config = dict(hidden_dims=[512, 512, 512], final_fc_init_scale=1e-2, is_out_dist=False)
        act_net_config = dict(act_f="gelu", use_dpout=False)
    elif spec.family == "antSoc":
        if spec.scale == "Ar":
            inv_m_config = dict(hidden_dims=[512, 512, 512], final_fc_init_scale=1e-2, is_out_dist=False)
        elif spec.scale == "Me":
            inv_m_config = dict(hidden_dims=[512, 1024, 1024, 512], final_fc_init_scale=1e-2, is_out_dist=False)
        else:
            raise NotImplementedError(f"antSoc scale {spec.scale}")
        act_net_config = dict(act_f="gelu", use_dpout=True, prob_dpout=0.2)
    elif spec.family == "humM":
        inv_m_config = dict(hidden_dims=[512, 1024, 1024, 512, 256], final_fc_init_scale=1e-2, is_out_dist=False)
        act_net_config = dict(act_f="gelu", use_dpout=True, prob_dpout=0.2)
    else:
        raise NotImplementedError(spec.family)

    return InvDynMLP(
        input_dim=input_dim, action_dim=act_dim, obs_dim=obs_dim,
        act_net_config=act_net_config, inv_m_config=inv_m_config
    )


def save_reference_plot(env_name: str, dataset: OgBInvDynDataset, normalizer, save_path: str, n_reference: int) -> None:
    """Render a few sampled training trajectories over the maze map for sanity checks."""
    if n_reference <= 0:
        return

    loader = torch.utils.data.DataLoader(dataset, batch_size=n_reference, shuffle=True, num_workers=0, pin_memory=True)
    obs_trajs, _act_trajs, _cond, _val_lens = next(iter(loader))
    obs_trajs = normalizer.unnormalize(obs_trajs, "observations").cpu().numpy()

    env = ogb_make_env(env_name)
    maze_map = env.unwrapped.maze_map

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(maze_map, cmap="gray_r", origin="upper")
    for traj in obs_trajs:
        traj_ij = ogb_xy_to_ij(env, traj[:, :2])
        ax.plot(traj_ij[:, 1], traj_ij[:, 0], linewidth=1.5, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])

    mkdir(os.path.dirname(save_path))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    try:
        env.close()
    except Exception:
        pass


def train_invdyn(args, spec: EnvSpec):
    """Train the inverse-dynamics model for one environment and checkpoint it."""
    device = args.device
    inv_name = args.invdyn_name
    inv_dir = os.path.join("logs", spec.env_name, "invdyn", inv_name)
    mkdir(inv_dir)
    save_json(vars(args) | {"env_spec": asdict(spec)}, os.path.join(inv_dir, "args.json"))

    full_normalizer = make_normalizer(spec.env_name, obs_select_dim=None)
    dataset = OgBInvDynDataset(spec, full_normalizer)
    obs_dim = dataset.obs_dim
    act_dim = dataset.act_dim

    if args.goal_sel_idxs:
        goal_sel_idxs = tuple(int(x) for x in args.goal_sel_idxs)
    else:
        goal_sel_idxs = tuple(range(spec.goal_dim))  # matches configs (first 2 dims)

    inv_model = build_invdyn_model_for_env(
        spec,
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=len(goal_sel_idxs),
    ).to(device)
    ema = EMA(args.ema_decay)
    ema_model = build_invdyn_model_for_env(
        spec,
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=len(goal_sel_idxs),
    ).to(device)
    ema_model.load_state_dict(inv_model.state_dict())

    opt = torch.optim.Adam(inv_model.parameters(), lr=args.lr)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        inv_model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        step = int(ckpt["step"])
        print(f"[invdyn] resumed from {args.resume} (step={step})")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )

    if args.n_reference > 0:
        save_reference_plot(
            env_name=spec.env_name,
            dataset=dataset,
            normalizer=full_normalizer,
            save_path=os.path.join(inv_dir, "_sample-reference.png"),
            n_reference=int(args.n_reference),
        )

    label_freq = max(1, int(args.n_train_steps // args.n_saves))
    t_start = time.time()

    while step < args.n_train_steps:
        for batch in loader:
            if step >= args.n_train_steps:
                break

            inv_model.train()
            for _ in range(args.grad_accum):
                obs_trajs, act_trajs, _cond, val_lens = batch
                obs_trajs = to_device(obs_trajs, device)
                act_trajs = to_device(act_trajs, device)
                val_lens = to_device(val_lens, device).long()

                B, H, _ = obs_trajs.shape
                goal_idxs = torch.randint(low=1, high=H, size=(B,), device=device)
                goal_idxs = torch.minimum(goal_idxs, val_lens - 1)

                b_idxs = torch.arange(B, device=device)
                x_t = obs_trajs[b_idxs, 0]
                x_goal_full = obs_trajs[b_idxs, goal_idxs]
                x_goal = x_goal_full[:, list(goal_sel_idxs)]
                a_t = act_trajs[b_idxs, 0]

                loss, _ = inv_model.loss(x_t, x_goal, a_t)
                (loss / args.grad_accum).backward()

            opt.step()
            opt.zero_grad(set_to_none=True)

            if step < args.step_start_ema:
                ema_model.load_state_dict(inv_model.state_dict())
            elif (step % args.update_ema_every) == 0:
                ema.update_model_average(ema_model, inv_model)

            if (step % args.save_freq) == 0:
                label = (step // label_freq) * label_freq
                save_path = os.path.join(inv_dir, f"state_{label}.pt")
                torch.save({"step": step, "model": inv_model.state_dict(), "ema": ema_model.state_dict()}, save_path)
                print(f"[invdyn] saved {save_path}")

            if (step % args.log_freq) == 0:
                print(f"[invdyn] step={step} loss={loss.item():.6f} lr={get_lr(opt):.2e} elapsed={((time.time()-t_start)/3600):.2f}h")

            step += 1


def main():
    """CLI entry point for inverse-dynamics training."""
    p = argparse.ArgumentParser()
    p.add_argument("--spec_csv", type=str, default="ogb_env_spec.csv")
    p.add_argument("--env", type=str, default="antmaze-giant-stitch-v0", help="Environment name, must be a key in ogb_env_spec.csv")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--invdyn_name", type=str, default="baseline")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--step_start_ema", type=int, default=4000)
    p.add_argument("--update_ema_every", type=int, default=10)
    p.add_argument("--n_train_steps", type=int, default=2_000_000)
    p.add_argument("--save_freq", type=int, default=4000)
    p.add_argument("--n_saves", type=int, default=5)
    p.add_argument("--log_freq", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--goal_sel_idxs", type=int, nargs="+", default=None,
                   help="Optional list of indices from the observation to use as goals (defaults to spec.goal_dim range).")
    p.add_argument("--n_reference", type=int, default=40, help="Number of reference trajectories to render.")
    p.add_argument("--n_samples", type=int, default=10, help="Unused (kept for parity with ogb configs).")

    args = p.parse_args()

    # Load env spec
    specs = load_env_specs(args.spec_csv)
    if args.env not in specs:
        raise ValueError(f"--env {args.env} not found in {args.spec_csv}. Available: {list(specs.keys())}")
    spec = specs[args.env]
    print(f"[invdyn] loaded env spec for {args.env}: {spec}")

    train_invdyn(args, spec)


if __name__ == "__main__":
    main()
