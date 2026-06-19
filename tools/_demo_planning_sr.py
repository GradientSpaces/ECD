"""Planning-SR helpers"""

import os
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from ecd.dataset.ogbench import ogb_make_env, ogb_maze_unit, ogb_offset_x, ogb_offset_y

def maze_spec(env_name):
    """Returns the maze map, unit size, and offset for a given OGBench environment."""
    env = ogb_make_env(env_name); mz = np.asarray(env.unwrapped.maze_map)
    u = ogb_maze_unit(env); ox, oy = ogb_offset_x(env), ogb_offset_y(env)
    try: env.close()
    except Exception: pass
    return mz, float(u), float(ox), float(oy)

def _cell_idx(traj_xy, u, ox, oy):
    """Converts a trajectory of x-y positions to i-j cell indices in the maze map."""
    j = np.round((traj_xy[:, 0] + ox) / u).astype(int)
    i = np.round((traj_xy[:, 1] + oy) / u).astype(int)
    return i, j

def collision_fraction(traj_xy, mz, u, ox, oy):
    """Fraction of plan waypoints that fall inside a wall cell (or leave the maze). Lower is better."""
    i, j = _cell_idx(traj_xy, u, ox, oy)
    oob = (i < 0) | (i >= mz.shape[0]) | (j < 0) | (j >= mz.shape[1])
    ii, jj = np.clip(i, 0, mz.shape[0] - 1), np.clip(j, 0, mz.shape[1] - 1)
    in_wall = mz[ii, jj] != 0
    return float(np.mean(oob | in_wall))

def is_feasible(traj_xy, mz, u, ox, oy, goal_xy, goal_tol_cells=1.5, coll_tol=0.0):
    """A plan is feasible if it (essentially) stays in free space and ends near the goal.
    coll_tol = allowed fraction of waypoints touching a wall (0.0 = strict; ~0.02 tolerates corner clips)."""
    if collision_fraction(traj_xy, mz, u, ox, oy) > coll_tol:
        return False
    return np.linalg.norm(traj_xy[-1] - np.asarray(goal_xy)) < goal_tol_cells * u
