"""Maze grid/world coordinate conversion and BFS shortest-path utilities."""

from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np


def xy_to_ij(rank_context: Dict[str, Any], xy: np.ndarray) -> np.ndarray:
    """Convert world xy coordinates to fractional maze (i, j) grid coordinates."""
    maze_unit = float(rank_context["maze_unit"])
    offset_x = float(rank_context["offset_x"])
    offset_y = float(rank_context["offset_y"])
    i = (xy[..., 1] + offset_y + 0.5 * maze_unit) / maze_unit
    j = (xy[..., 0] + offset_x + 0.5 * maze_unit) / maze_unit
    return np.stack([i, j], axis=-1) - 0.5


def ij_to_xy(rank_context: Dict[str, Any], ij: np.ndarray) -> np.ndarray:
    """Convert maze (i, j) grid coordinates back to world xy coordinates."""
    maze_unit = float(rank_context["maze_unit"])
    offset_x = float(rank_context["offset_x"])
    offset_y = float(rank_context["offset_y"])
    x = ij[..., 1] * maze_unit - offset_x
    y = ij[..., 0] * maze_unit - offset_y
    return np.stack([x, y], axis=-1).astype(np.float32)


def nearest_free_cell(maze_map: np.ndarray, ij: np.ndarray) -> Tuple[int, int]:
    """Return the free maze cell closest to the given (i, j) coordinate."""
    free = np.argwhere(maze_map == 0)
    if len(free) == 0:
        raise ValueError("maze_map has no free cells")
    d2 = np.sum((free.astype(np.float32) - ij[None, :]) ** 2, axis=1)
    cell = free[int(np.argmin(d2))]
    return int(cell[0]), int(cell[1])


def shortest_cell_path(maze_map: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Return the BFS shortest path of free cells from ``start`` to ``goal``."""
    h, w = maze_map.shape
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        i, j = cur
        for nxt in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            ni, nj = nxt
            if 0 <= ni < h and 0 <= nj < w and maze_map[ni, nj] == 0 and nxt not in seen:
                seen.add(nxt)
                prev[nxt] = cur
                q.append(nxt)

    if goal not in seen:
        return [start, goal]
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def resample_polyline(points_xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a 2D polyline to ``n`` points evenly spaced by arc length."""
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if len(points_xy) == 0:
        raise ValueError("points_xy is empty")
    if len(points_xy) == 1 or n <= 1:
        return np.repeat(points_xy[:1], max(1, n), axis=0)

    seg = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(dist[-1])
    if total <= 1e-8:
        return np.repeat(points_xy[:1], n, axis=0)
    target = np.linspace(0.0, total, int(n), dtype=np.float32)
    out = np.empty((int(n), 2), dtype=np.float32)
    out[:, 0] = np.interp(target, dist, points_xy[:, 0])
    out[:, 1] = np.interp(target, dist, points_xy[:, 1])
    return out


def shortest_path_plan_xy(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    rank_context: Dict[str, Any],
    n_points: int,
) -> np.ndarray:
    """Plan a ground-truth maze shortest path between two world xy points, resampled to ``n_points``."""
    maze_map = np.asarray(rank_context["maze_map"])
    st_ij = xy_to_ij(rank_context, np.asarray(start_xy, dtype=np.float32))
    gl_ij = xy_to_ij(rank_context, np.asarray(goal_xy, dtype=np.float32))
    st_cell = nearest_free_cell(maze_map, st_ij)
    gl_cell = nearest_free_cell(maze_map, gl_ij)
    cells = np.asarray(shortest_cell_path(maze_map, st_cell, gl_cell), dtype=np.float32)
    cell_xy = ij_to_xy(rank_context, cells)
    points = np.concatenate([start_xy[None, :2].astype(np.float32), cell_xy, goal_xy[None, :2].astype(np.float32)], axis=0)
    out = resample_polyline(points, n_points)
    out[0] = start_xy[:2]
    out[-1] = goal_xy[:2]
    return out
