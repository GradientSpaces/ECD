"""OGBench environment construction, dataset loading, and normalizer helpers."""

from typing import Tuple, List, Optional

import numpy as np
import torch

from .common import EnvSpec, DatasetNormalizer


# OGBench-related utilities

def resolve_ogb_env_name(env_name: str) -> str:
    """Map local planner aliases to actual OGBench environment names."""
    alias = {
        "antmaze-medium-stitch-v0-o15d": "antmaze-medium-stitch-v0",
        "antmaze-large-stitch-v0-o15d": "antmaze-large-stitch-v0",
    }
    return alias.get(env_name, env_name)

def ogb_make_env(env_name: str, **kwargs):
    """Build an OGBench environment (env only), tagging it with local/resolved names."""
    import ogbench
    resolved_name = resolve_ogb_env_name(env_name)
    wrapped = ogbench.make_env_and_datasets(resolved_name, env_only=True, **kwargs)
    env = wrapped.unwrapped
    env.max_episode_steps = wrapped._max_episode_steps
    # Keep the actual OGBench name for dataset loading; store the local alias separately.
    env.name = resolved_name
    env.local_name = env_name
    return env


def ogb_load_train_dataset(env):
    """Load the OGBench training dataset associated with ``env``."""
    import ogbench
    train_dataset, _ = ogbench.make_env_and_datasets(
        env.name, dataset_only=True, cur_env=env
    )
    return train_dataset


def ogb_segment_episodes(obs: np.ndarray, act: np.ndarray, terminals: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split flat (obs, act) arrays into per-episode lists using ``terminals``."""
    assert obs.shape[0] == act.shape[0] == terminals.shape[0]
    term_idxs = np.where(terminals.astype(np.bool_))[0] + 1
    obs_eps = np.split(obs, term_idxs, axis=0)
    act_eps = np.split(act, term_idxs, axis=0)
    out = []
    for o, a in zip(obs_eps, act_eps):
        if len(o) == 0:
            continue
        out.append((o.astype(np.float32), a.astype(np.float32)))
    return out


def compute_norm_stats_from_ogbench(env_name: str):
    """Compute per-dimension obs/act min and max from an env's training dataset."""
    env = ogb_make_env(env_name)
    d = ogb_load_train_dataset(env)
    obs = d["observations"].astype(np.float32)
    act = d["actions"].astype(np.float32)
    try:
        env.close()
    except Exception:
        pass
    return obs.min(0), obs.max(0), act.min(0), act.max(0)


def make_normalizer(env_name: str, obs_select_dim: Optional[Tuple[int, ...]]):
    """
    Returns DatasetNormalizer.
    - If hardcoded stats exist: use them.
    - Else if require_hardcoded: raise.
    - Else: compute from dataset and cache to npz.
    """
    stats = get_hardcoded_norm_stats(env_name)
    obs_min, obs_max, act_min, act_max = compute_norm_stats_from_ogbench(env_name)

    if stats is None:
        print(f"[norm] computed obs_min: {obs_min}, obs_max: {obs_max}")
        print(f"[norm] computed act_min: {act_min}, act_max: {act_max}")
    else:
        # do a sanity check
        h_obs_min, h_obs_max, h_act_min, h_act_max = stats
        assert np.allclose(obs_min, h_obs_min), f"obs_min mismatch for {env_name}, obs_min: {obs_min}, h_obs_min: {h_obs_min}"
        assert np.allclose(obs_max, h_obs_max), f"obs_max mismatch for {env_name}, obs_max: {obs_max}, h_obs_max: {h_obs_max}"
        assert np.allclose(act_min, h_act_min), f"act_min mismatch for {env_name}, act_min: {act_min}, h_act_min: {h_act_min}"
        assert np.allclose(act_max, h_act_max), f"act_max mismatch for {env_name}, act_max: {act_max}, h_act_max: {h_act_max}"
        obs_min, obs_max, act_min, act_max = stats

    if obs_select_dim is None:
        obs_mins = obs_min
        obs_maxs = obs_max
    else:
        obs_mins = obs_min[list(obs_select_dim)]
        obs_maxs = obs_max[list(obs_select_dim)]
    return DatasetNormalizer(obs_mins, obs_maxs, act_min, act_max)


def ogb_maze_unit(env) -> float:
    """Maze unit size, compatible across OGBench versions (getter or private attr)."""
    env = env.unwrapped
    if hasattr(env, "get_maze_unit"):
        return float(env.get_maze_unit())
    return float(env._maze_unit)


def ogb_offset_x(env) -> float:
    """Maze x offset, compatible across OGBench versions (getter or private attr)."""
    env = env.unwrapped
    if hasattr(env, "get_offset_x"):
        return float(env.get_offset_x())
    return float(env._offset_x)


def ogb_offset_y(env) -> float:
    """Maze y offset, compatible across OGBench versions (getter or private attr)."""
    env = env.unwrapped
    if hasattr(env, "get_offset_y"):
        return float(env.get_offset_y())
    return float(env._offset_y)


def ogb_xy_to_ij(env, xy: np.ndarray) -> np.ndarray:
    """
    Convert OGB mujoco xy to grid ij consistent with maze_map.
    xy: (...,2) array in mujoco coordinates.
    """
    # if xy.shape[-1] != 2, we assume it's first two dims are xy
    maze_unit = ogb_maze_unit(env)
    i = (xy[..., 1] + ogb_offset_y(env) + 0.5 * maze_unit) / maze_unit
    j = (xy[..., 0] + ogb_offset_x(env) + 0.5 * maze_unit) / maze_unit
    out = np.stack([i, j], axis=-1) - 0.5
    return out


def get_hardcoded_norm_stats(env_name: str):
    """
    Returns (obs_min, obs_max, act_min, act_max) or None if not available.
    For humanoidmaze, left as placeholder (user will fill later).
    """
    env_name = resolve_ogb_env_name(env_name)
    from .ogb_const import (
        # Ant maze stats
        OgB_AntMaze_Giant_Stitch_Obs_Min, OgB_AntMaze_Giant_Stitch_Obs_Max,
        OgB_AntMaze_Large_Stitch_Obs_Min, OgB_AntMaze_Large_Stitch_Obs_Max,
        OgB_AntMaze_Medium_Stitch_Obs_Min, OgB_AntMaze_Medium_Stitch_Obs_Max,
        OgB_AntMaze_Act_Min, OgB_AntMaze_Act_Max,
        # Antsoccer maze stats
        OgB_AntSoccer_Arena_Stitch_Obs_Min, OgB_AntSoccer_Arena_Stitch_Obs_Max,
        OgB_AntSoccer_Medium_Stitch_Obs_Min, OgB_AntSoccer_Medium_Stitch_Obs_Max,
        # Humanoid maze stats
        OgB_HumanoidMaze_Giant_Stitch_Obs_Min, OgB_HumanoidMaze_Giant_Stitch_Obs_Max,
        OgB_HumanoidMaze_Large_Stitch_Obs_Min, OgB_HumanoidMaze_Large_Stitch_Obs_Max,
        OgB_HumanoidMaze_Medium_Stitch_Obs_Min, OgB_HumanoidMaze_Medium_Stitch_Obs_Max,
        OgB_HumanoidMaze_Giant_Act_Min, OgB_HumanoidMaze_Giant_Act_Max,
        # Point maze stats
        OgB_PointMaze_Medium_Stitch_Obs_Min, OgB_PointMaze_Medium_Stitch_Obs_Max,
        OgB_PointMaze_Large_Stitch_Obs_Min, OgB_PointMaze_Large_Stitch_Obs_Max,
        OgB_PointMaze_Giant_Stitch_Obs_Min, OgB_PointMaze_Giant_Stitch_Obs_Max,
        OgB_PointMaze_Giant_Act_Min, OgB_PointMaze_Giant_Act_Max
    )

    if env_name == "antmaze-giant-stitch-v0":
        return OgB_AntMaze_Giant_Stitch_Obs_Min, OgB_AntMaze_Giant_Stitch_Obs_Max, OgB_AntMaze_Act_Min, OgB_AntMaze_Act_Max
    if env_name == "antmaze-large-stitch-v0":
        return OgB_AntMaze_Large_Stitch_Obs_Min, OgB_AntMaze_Large_Stitch_Obs_Max, OgB_AntMaze_Act_Min, OgB_AntMaze_Act_Max
    if env_name == "antmaze-medium-stitch-v0":
        return OgB_AntMaze_Medium_Stitch_Obs_Min, OgB_AntMaze_Medium_Stitch_Obs_Max, OgB_AntMaze_Act_Min, OgB_AntMaze_Act_Max
    if env_name == "humanoidmaze-giant-stitch-v0":
        return OgB_HumanoidMaze_Giant_Stitch_Obs_Min, OgB_HumanoidMaze_Giant_Stitch_Obs_Max, OgB_HumanoidMaze_Giant_Act_Min, OgB_HumanoidMaze_Giant_Act_Max
    if env_name == "humanoidmaze-large-stitch-v0":
        return OgB_HumanoidMaze_Large_Stitch_Obs_Min, OgB_HumanoidMaze_Large_Stitch_Obs_Max, OgB_HumanoidMaze_Giant_Act_Min, OgB_HumanoidMaze_Giant_Act_Max
    if env_name == "humanoidmaze-medium-stitch-v0":
        return OgB_HumanoidMaze_Medium_Stitch_Obs_Min, OgB_HumanoidMaze_Medium_Stitch_Obs_Max, OgB_HumanoidMaze_Giant_Act_Min, OgB_HumanoidMaze_Giant_Act_Max
    if env_name == "antsoccer-arena-stitch-v0":
        return OgB_AntSoccer_Arena_Stitch_Obs_Min, OgB_AntSoccer_Arena_Stitch_Obs_Max, OgB_AntMaze_Act_Min, OgB_AntMaze_Act_Max
    if env_name == "antsoccer-medium-stitch-v0":
        return OgB_AntSoccer_Medium_Stitch_Obs_Min, OgB_AntSoccer_Medium_Stitch_Obs_Max, OgB_AntMaze_Act_Min, OgB_AntMaze_Act_Max
    if env_name == "pointmaze-medium-stitch-v0":
        return OgB_PointMaze_Medium_Stitch_Obs_Min, OgB_PointMaze_Medium_Stitch_Obs_Max, OgB_PointMaze_Giant_Act_Min, OgB_PointMaze_Giant_Act_Max
    if env_name == "pointmaze-large-stitch-v0":
        return OgB_PointMaze_Large_Stitch_Obs_Min, OgB_PointMaze_Large_Stitch_Obs_Max, OgB_PointMaze_Giant_Act_Min, OgB_PointMaze_Giant_Act_Max
    if env_name == "pointmaze-giant-stitch-v0":
        return OgB_PointMaze_Giant_Stitch_Obs_Min, OgB_PointMaze_Giant_Stitch_Obs_Max, OgB_PointMaze_Giant_Act_Min, OgB_PointMaze_Giant_Act_Max
    
    
    return None


# Dataset classes

class OgBPlanningDataset(torch.utils.data.Dataset):
    """
    Planning dataset (like OgB_SeqDataset_V2) for obs_select_dim planner training:
      - pad_option_2='buf', pad_type='first', extra_pad=64 by default.
      - returns (obs_trajs, act_trajs, cond_st_gl) where obs_trajs are normalized.
    """

    def __init__(
        self,
        spec: EnvSpec,
        normalizer: DatasetNormalizer,
    ):
        super().__init__()
        self.spec = spec
        self.horizon = spec.plan_sm_horizon
        self.obs_select_dim = spec.plan_obs_select_dim
        self.extra_pad = spec.plan_extra_pad
        self.max_path_length = spec.plan_max_path_length
        self.max_ori_path_len = spec.plan_max_ori_path_len
        self.normalizer = normalizer

        env = ogb_make_env(spec.env_name)
        dset = ogb_load_train_dataset(env)
        episodes = ogb_segment_episodes(dset["observations"], dset["actions"], dset["terminals"])
        try:
            env.close()
        except Exception:
            pass

        obs_list = []
        act_list = []
        lengths = []

        for obs_ep, act_ep in episodes:
            obs_ep = obs_ep[:, list(self.obs_select_dim)]
            L = obs_ep.shape[0]
            if spec.plan_use_padbuf and self.extra_pad > 0:
                obs_pad = np.repeat(obs_ep[:1], repeats=self.extra_pad, axis=0)
                act_pad = np.zeros((self.extra_pad, act_ep.shape[1]), dtype=np.float32)
                obs_ep = np.concatenate([obs_pad, obs_ep], axis=0)
                act_ep = np.concatenate([act_pad, act_ep], axis=0)

            obs_nm = self.normalizer.normalize(obs_ep, "observations")
            act_nm = self.normalizer.normalize(act_ep, "actions")

            obs_list.append(torch.from_numpy(obs_nm).float())
            act_list.append(torch.from_numpy(act_nm).float())
            lengths.append(obs_nm.shape[0])

        self.obs_eps = obs_list
        self.act_eps = act_list
        self.path_lengths = np.asarray(lengths, dtype=np.int32)

        indices = []
        for ep_idx, L in enumerate(self.path_lengths.tolist()):
            if self.max_ori_path_len is not None:
                pad_extra = self.extra_pad if spec.plan_use_padbuf else 0
                L = min(L, self.max_ori_path_len + pad_extra)
            L = min(L, self.max_path_length)
            max_start = min(L - 1, self.max_path_length - self.horizon)
            max_start = min(max_start, L - self.horizon)
            for s in range(max_start + 1):
                indices.append((ep_idx, s))
        self.indices = np.asarray(indices, dtype=np.int32)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        ep_idx, start = self.indices[int(idx)]
        end = start + self.horizon
        obs = self.obs_eps[ep_idx][start:end]
        act = self.act_eps[ep_idx][start:end]
        cond = {0: obs[0].clone(), self.horizon - 1: obs[-1].clone()}
        return obs, act, cond


class OgBInvDynDataset(torch.utils.data.Dataset):
    """
    Inverse dynamics dataset (like OgB_InvDyn_SeqDataset_V1):
      - returns sequences of length invdyn_horizon with zero padding beyond episode length.
      - also returns val_len for lookahead sampling.
    """

    def __init__(self, spec: EnvSpec, normalizer: DatasetNormalizer):
        super().__init__()
        self.spec = spec
        self.horizon = spec.invdyn_horizon
        self.max_path_length = spec.invdyn_max_path_length
        self.normalizer = normalizer

        env = ogb_make_env(spec.env_name)
        dset = ogb_load_train_dataset(env)
        episodes = ogb_segment_episodes(dset["observations"], dset["actions"], dset["terminals"])
        try:
            env.close()
        except Exception:
            pass

        self.obs_dim = episodes[0][0].shape[1]
        self.act_dim = episodes[0][1].shape[1]

        n_eps = len(episodes)
        obs_buf = np.zeros((n_eps, self.max_path_length, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((n_eps, self.max_path_length, self.act_dim), dtype=np.float32)
        path_lengths = np.zeros((n_eps,), dtype=np.int32)

        for i, (obs_ep, act_ep) in enumerate(episodes):
            L = obs_ep.shape[0]
            assert L <= self.max_path_length
            obs_buf[i, :L] = obs_ep
            act_buf[i, :L] = act_ep
            path_lengths[i] = L

        obs_nm = self.normalizer.normalize(obs_buf, "observations")
        act_nm = self.normalizer.normalize(act_buf, "actions")

        self.obs_nm = torch.from_numpy(obs_nm).float()
        self.act_nm = torch.from_numpy(act_nm).float()
        self.path_lengths = path_lengths

        indices = []
        for ep_idx, L in enumerate(self.path_lengths.tolist()):
            max_start = (L - 2)
            for s in range(max_start + 1):
                val_len = min(L - s, self.horizon)
                indices.append((ep_idx, s, val_len))
        self.indices = np.asarray(indices, dtype=np.int32)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        ep_idx, start, val_len = self.indices[int(idx)]
        end = start + self.horizon
        obs = self.obs_nm[ep_idx, start:end]
        act = self.act_nm[ep_idx, start:end]
        conditions = torch.zeros((), dtype=torch.int32)
        return obs, act, conditions, torch.tensor(int(val_len), dtype=torch.int32)
