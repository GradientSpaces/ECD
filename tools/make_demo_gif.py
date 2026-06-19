"""Teaser animation code.

Left:  100 short trajectory fragments sampled from the offline dataset -- all the model is ever
       trained on (short clips; it never sees a full giant-maze traversal).
Mid/Right: for one start->goal pair, each planner produces N_PLANS plans. Every plan is the planner's
       real output: sample B_S candidates, rank them (map-free), and blend the top ones into a single
       stitched trajectory (as at eval time, but with no replanning). A plan fails if it touches a
       wall at any step ("one-shot success"). ECD's boundary-consistent stitch threads the maze where
       CD's heuristic stitch smears plans through walls. Plan-only: no actions, no inverse dynamics.

Run:      python tools/make_demo_gif.py
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.getcwd())

import numpy as np, torch, matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.animation import FuncAnimation, PillowWriter
from tools._demo_core import load_planner, make_policy, plan_xy
from tools._demo_planning_sr import maze_spec, is_feasible
from ecd.dataset.ogbench import ogb_make_env, ogb_load_train_dataset, ogb_segment_episodes

ENV = "antmaze-giant-stitch-v0"
ENV_LABEL = "AntMaze-giant-stitch"
N_COMP = 9

PROB_IDX = int(os.environ.get("PROB_IDX", 0))
B_S = int(os.environ.get("B_S", 40))   # candidate pool ranked + blended into ONE plan (no replanning)
N_PLANS = 24        # plans drawn per planner
N_FRAG = 100        # training fragments shown
N_FRAMES = 80; FPS = 16; HOLD_SEC = 2.0; SEED = 0; GOAL_TOL = 2.0
WALL_C = "#e7eaf1"                                                  # soft maze walls
SUCCESS_C = "#2563eb"                                              # wall-clean plans: blue for both planners (fair)
FAIL_C = "#ef3b2c"                                                  # plans that hit a wall: vivid, clearly visible
FRAG_C = "#9aa3b2"
ECD_CFG = dict(base_scale=0.5, react_scale=0.1, react_clip=1.0,     # released antmaze-giant recipe
               chunk_react_type="markov", markov_type="laplacian", markov_rho=0.25)

def resample(path, n):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if d[-1] == 0: return np.repeat(path[:1], n, axis=0)
    return np.stack([np.interp(np.linspace(0, d[-1], n), d, path[:, k]) for k in range(2)], axis=1)

def hold_last_frame(path, fps, hold_sec):
    """Make the last frame linger ~hold_sec: Pillow merges duplicate frames, so we lengthen the final
    frame's per-frame duration instead of appending copies."""
    from PIL import Image, ImageSequence
    frames = [f.copy().convert("RGB") for f in ImageSequence.Iterator(Image.open(path))]
    dur = [int(1000 / fps)] * (len(frames) - 1) + [int(hold_sec * 1000)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0, disposal=2)

def style(ax):
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.02)
    for sp in ax.spines.values(): sp.set_edgecolor("#d7dbe3"); sp.set_linewidth(1.0)

def draw_walls(ax, mz, u, ox, oy):
    for i in range(mz.shape[0]):
        for j in range(mz.shape[1]):
            if mz[i, j] != 0:
                ax.add_patch(plt.Rectangle((j*u-ox-u/2, i*u-oy-u/2), u, u, color=WALL_C, lw=0))

def load_fragments(n):
    env = ogb_make_env(ENV); ds = ogb_load_train_dataset(env)
    eps = ogb_segment_episodes(ds["observations"], ds["actions"], ds["terminals"])
    pick = np.random.default_rng(SEED).choice(len(eps), size=min(n, len(eps)), replace=False)
    try: env.close()
    except Exception: pass
    return [eps[k][0][:, :2] for k in pick]

def sample_panel(policy, s, g, mz, u, ox, oy, b_s):
    plans = [np.asarray(plan_xy(policy, s, g, b_size=b_s)) for _ in range(N_PLANS)]
    ok = np.array([bool(is_feasible(p, mz, u, ox, oy, g, goal_tol_cells=GOAL_TOL, coll_tol=0.0)) for p in plans])
    return dict(rs=np.stack([resample(p, N_FRAMES) for p in plans]), ok=ok, psr=float(ok.mean()))

class _FragHandler(HandlerBase):
    """Legend icon for a training fragment: a line with a start (o) and end (square) marker."""
    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        y = h / 2.0 - yd
        return [Line2D([0, w], [y, y], color=FRAG_C, lw=2.0, transform=trans),
                Line2D([w*0.06], [y], marker="o", mfc=FRAG_C, mec="white", mew=0.7, ms=10, ls="", transform=trans),
                Line2D([w*0.94], [y], marker="s", mfc=FRAG_C, mec="white", mew=0.7, ms=9, ls="", transform=trans)]
class _FragKey: pass

def _plans_for(cd_policy, ecd_policy, probs, mz, u, ox, oy, b_s, prob_idx):
    """Sample (or load from cache) the fragments and the two planners' plans for one start->goal pair."""
    cache = f"/tmp/gif_cache_bs{b_s}_p{prob_idx}.npz"
    if os.path.exists(cache) and not os.environ.get("NOCACHE"):
        z = np.load(cache, allow_pickle=True)
        panels = {"CompDiffuser (CD)": dict(rs=z["cd_rs"], ok=z["cd_ok"], psr=float(z["cd_psr"])),
                  "ECD (Ours)":        dict(rs=z["ecd_rs"], ok=z["ecd_ok"], psr=float(z["ecd_psr"]))}
        return list(z["frags"]), panels, z["s"], z["g"]
    s, g = probs["start_state"][prob_idx][:2], probs["goal_pos"][prob_idx][:2]
    frags = load_fragments(N_FRAG)
    panels = {}
    for name, pol in [("CompDiffuser (CD)", cd_policy), ("ECD (Ours)", ecd_policy)]:
        torch.manual_seed(SEED); panels[name] = sample_panel(pol, s, g, mz, u, ox, oy, b_s)
        print(f"{name}: one-shot success = {panels[name]['psr']*100:.0f}%  ({int(panels[name]['ok'].sum())}/{N_PLANS})", flush=True)
    np.savez(cache, frags=np.array(frags), s=s, g=g,
             cd_rs=panels["CompDiffuser (CD)"]["rs"], cd_ok=panels["CompDiffuser (CD)"]["ok"], cd_psr=panels["CompDiffuser (CD)"]["psr"],
             ecd_rs=panels["ECD (Ours)"]["rs"], ecd_ok=panels["ECD (Ours)"]["ok"], ecd_psr=panels["ECD (Ours)"]["psr"])
    return frags, panels, s, g

def _draw(frags, panels, s, g, mz, u, ox, oy, b_s, out):
    fig, (ax_tr, ax_cd, ax_ecd) = plt.subplots(1, 3, figsize=(16.5, 6.4)); fig.patch.set_facecolor("white")

    draw_walls(ax_tr, mz, u, ox, oy)                                   # left: training fragments
    cmap = plt.cm.turbo(np.linspace(0.04, 0.96, len(frags))); np.random.default_rng(1).shuffle(cmap)
    for c, fr in zip(cmap, frags):
        ax_tr.plot(fr[:, 0], fr[:, 1], color=c, lw=0.9, alpha=0.75, zorder=4)
        ax_tr.plot(fr[0, 0], fr[0, 1], "o", ms=4.5, mfc=c, mec="white", mew=0.6, zorder=5)
        ax_tr.plot(fr[-1, 0], fr[-1, 1], "s", ms=4.5, mfc=c, mec="white", mew=0.6, zorder=5)
    style(ax_tr)
    ax_tr.text(0.5, 1.115, "Training fragments", transform=ax_tr.transAxes, ha="center", va="bottom",
               fontsize=22, fontweight="bold", color="#0f172a")
    ax_tr.text(0.5, 1.018, ENV_LABEL, transform=ax_tr.transAxes, ha="center", va="bottom",
               fontsize=17, family="monospace", color="#475569")

    lcs = {}                                                           # mid/right: CD and ECD plans (animated)
    for ax, name in [(ax_cd, "CompDiffuser (CD)"), (ax_ecd, "ECD (Ours)")]:
        d = panels[name]; draw_walls(ax, mz, u, ox, oy)
        ax.plot(*s, marker="o", ms=11, mfc="#111827", mec="white", mew=1.6, zorder=7)
        ax.plot(*g, marker="*", ms=24, mfc="#fbbf24", mec="#111827", mew=1.3, zorder=7)
        lcs[name] = [ax.add_collection(LineCollection([], colors=[SUCCESS_C if ok else FAIL_C],
                        linewidths=(2.2 if ok else 1.8), alpha=(0.85 if ok else 0.6),
                        capstyle="round", zorder=(5 if ok else 4))) for ok in d["ok"]]
        style(ax)
        is_ecd = (name == "ECD (Ours)")    # highlight ours; CD stays a neutral gray, not bold
        ax.text(0.5, 1.115, name, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=22, fontweight="bold", color="#0f172a")
        ax.text(0.5, 1.018, f"One-shot success: {d['psr']*100:.0f}%", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=17, fontweight=("bold" if is_ecd else "normal"),
                color=(SUCCESS_C if is_ecd else "#6b7280"))

    handles = [_FragKey(),
               Line2D([0],[0], marker="o", color="w", mfc="#111827", mec="white", ms=11),
               Line2D([0],[0], marker="*", color="w", mfc="#fbbf24", mec="#111827", ms=17),
               Line2D([0],[0], color=SUCCESS_C, lw=3.5),
               Line2D([0],[0], color=FAIL_C, lw=3.5)]
    labels = ["Training fragment (each with a unique color)", "Start", "Goal",
              "Plan reaches goal, wall-free", "Plan hits a wall"]
    fig.legend(handles, labels, handler_map={_FragKey: _FragHandler()}, loc="lower center", ncol=5,
               frameon=False, fontsize=15.5, handletextpad=0.6, columnspacing=2.2, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.008, right=0.992, top=0.85, bottom=0.105, wspace=0.04)

    def update(f):
        arts = []
        for name in ("CompDiffuser (CD)", "ECD (Ours)"):
            for lc, path in zip(lcs[name], panels[name]["rs"]):
                lc.set_segments([path[:f+1]]); arts.append(lc)
        return arts

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    update(N_FRAMES - 1); fig.savefig("/tmp/gif_preview.png", dpi=110)
    if os.environ.get("PREVIEW_ONLY"):
        print("preview only -> /tmp/gif_preview.png", flush=True); plt.close(fig); return out
    FuncAnimation(fig, update, frames=list(range(N_FRAMES)), blit=False).save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    hold_last_frame(out, FPS, HOLD_SEC)   # freeze the final frame ~HOLD_SEC seconds
    return out

def render_teaser(cd_policy, ecd_policy, probs, env=ENV, b_s=B_S, prob_idx=PROB_IDX, out=None):
    """Render the 1x3 teaser gif from already-built CD and ECD policies. Returns the output path."""
    out = out or ("assets/ecd_vs_cd_giant.gif" if b_s == 40 else f"assets/ecd_vs_cd_giant_bs{b_s}.gif")
    mz, u, ox, oy = maze_spec(env)
    frags, panels, s, g = _plans_for(cd_policy, ecd_policy, probs, mz, u, ox, oy, b_s, prob_idx)
    path = _draw(frags, panels, s, g, mz, u, ox, oy, b_s, out)
    print("wrote", path, flush=True)
    return path

def main():
    spec, normalizer, diffusion = load_planner(ENV, device="cuda")
    cd  = make_policy(diffusion, normalizer, N_COMP, infer_type="interleave", rank_type="overlap")
    ecd = make_policy(diffusion, normalizer, N_COMP, infer_type="ecd_chunk", rank_type="overlap", **ECD_CFG)
    from ecd.eval import load_eval_problems
    probs = load_eval_problems(spec.eval_probs_h5)
    render_teaser(cd, ecd, probs)

if __name__ == "__main__":
    main()
