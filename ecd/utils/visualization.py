"""Matplotlib helpers for rendering maze maps, plans, and rollout trajectories."""

import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from ..dataset.ogbench import ogb_xy_to_ij
from .io import mkdir


def plot_maze_and_traj(ax, maze_map: np.ndarray, traj_ij: np.ndarray, color: str, label: str, start_ij=None, goal_ij=None):
    """Draw a maze map plus a single trajectory (with optional start/goal markers) on ``ax``."""
    ax.imshow(maze_map, cmap="gray_r", origin="upper")
    if traj_ij is not None and len(traj_ij) > 0:
        ax.plot(traj_ij[:, 1], traj_ij[:, 0], color=color, linewidth=2, label=label)
    if start_ij is not None:
        ax.scatter([start_ij[1]], [start_ij[0]], c="white", s=60, marker="o", edgecolors="k", linewidths=1)
    if goal_ij is not None:
        ax.scatter([goal_ij[1]], [goal_ij[0]], c="gold", s=80, marker="*", edgecolors="k", linewidths=1)
    ax.set_xticks([])
    ax.set_yticks([])


def save_training_validation_plot(
    save_path: str,
    env,
    starts_xy: np.ndarray,
    goals_xy: np.ndarray,
    plans_interleave: List[np.ndarray],
):
    """Save a 3-panel plot of the interleave plans for validation."""
    maze_map = env.unwrapped.maze_map
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for i in range(3):
        st_ij = ogb_xy_to_ij(env, starts_xy[i])
        gl_ij = ogb_xy_to_ij(env, goals_xy[i])
        tj_i = ogb_xy_to_ij(env, plans_interleave[i])

        ax = axes[i]
        plot_maze_and_traj(ax, maze_map, tj_i, "tab:blue", "interleave", start_ij=st_ij, goal_ij=gl_ij)
        # put the legend below the plot
        ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1, -0.1))

    mkdir(os.path.dirname(save_path))
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[val_vis] {save_path}")


def save_eval_samples_plot(
    save_path: str,
    env,
    starts_xy: List[np.ndarray],
    goals_xy: List[np.ndarray],
    plan_xy_list: List[np.ndarray],
    rollout_xy_list: List[np.ndarray],
):
    """Save a 2x5 grid of plan vs. rollout trajectories for the first eval episodes."""
    import matplotlib.pyplot as plt

    maze_map = env.unwrapped.maze_map
    n = min(5, len(plan_xy_list))
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    for i in range(5):
        ax_plan = axes[0, i]
        ax_roll = axes[1, i]
        if i >= n:
            ax_plan.axis("off")
            ax_roll.axis("off")
            continue

        st_ij = ogb_xy_to_ij(env, starts_xy[i])
        gl_ij = ogb_xy_to_ij(env, goals_xy[i])

        plan_ij = ogb_xy_to_ij(env, plan_xy_list[i])
        roll_ij = ogb_xy_to_ij(env, rollout_xy_list[i])

        plot_maze_and_traj(ax_plan, maze_map, plan_ij, "tab:blue", "plan", start_ij=st_ij, goal_ij=gl_ij)
        ax_plan.set_title(f"Plan {i}", fontsize=10)

        plot_maze_and_traj(ax_roll, maze_map, roll_ij, "tab:green", "rollout", start_ij=st_ij, goal_ij=gl_ij)
        ax_roll.set_title(f"Rollout {i}", fontsize=10)

    mkdir(os.path.dirname(save_path))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[eval_vis] {save_path}")


def save_eval_tracking_plot(
    save_path: str,
    env,
    starts_xy: List[np.ndarray],
    goals_xy: List[np.ndarray],
    plan_xy_list: List[np.ndarray],
    rollout_xy_list: List[np.ndarray],
    tracking_dist_list: List[np.ndarray],
):
    """Save a 1x5 plot of plan vs. rollout with per-step tracking-error connectors."""
    maze_map = env.unwrapped.maze_map
    n = min(5, len(plan_xy_list))
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))

    for i in range(5):
        ax = axes[i]
        if i >= n:
            ax.axis("off")
            continue

        st_ij = ogb_xy_to_ij(env, starts_xy[i])
        gl_ij = ogb_xy_to_ij(env, goals_xy[i])
        plan_ij = ogb_xy_to_ij(env, plan_xy_list[i])
        roll_ij = ogb_xy_to_ij(env, rollout_xy_list[i])

        ax.imshow(maze_map, cmap="gray_r", origin="upper")
        if len(plan_ij) > 0:
            ax.plot(plan_ij[:, 1], plan_ij[:, 0], color="tab:blue", linewidth=1.8, label="plan")
        if len(roll_ij) > 0:
            ax.plot(roll_ij[:, 1], roll_ij[:, 0], color="tab:green", linewidth=1.8, label="rollout")

        stride = max(1, len(plan_ij) // 24)
        for j in range(0, min(len(plan_ij), len(roll_ij)), stride):
            ax.plot(
                [roll_ij[j, 1], plan_ij[j, 1]],
                [roll_ij[j, 0], plan_ij[j, 0]],
                color="tab:red",
                linewidth=0.5,
                alpha=0.35,
            )

        d = tracking_dist_list[i] if i < len(tracking_dist_list) else np.asarray([], dtype=np.float32)
        if len(d) > 0:
            title = f"Track {i}  mean={float(np.mean(d)):.2f} p90={float(np.percentile(d, 90)):.2f}"
        else:
            title = f"Track {i}"
        ax.set_title(title, fontsize=9)
        ax.scatter([st_ij[1]], [st_ij[0]], c="white", s=60, marker="o", edgecolors="k", linewidths=1)
        ax.scatter([gl_ij[1]], [gl_ij[0]], c="gold", s=80, marker="*", edgecolors="k", linewidths=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")

    mkdir(os.path.dirname(save_path))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[track_vis] {save_path}")

