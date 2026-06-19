"""Environment specs (loaded from CSV) and dataset normalization classes."""

from dataclasses import dataclass
import csv
import ast
from typing import Dict, Tuple, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class EnvSpec:
    """Per-environment planner/invdyn/eval configuration parsed from a CSV row."""


    env_name: str
    family: str           # antM | humM
    scale: str            # Gi | Lg | Me
    train_traj_len: int
    obs_dim_full: int
    goal_dim: int

    plan_obs_select_dim: Tuple[int, ...]
    plan_sm_horizon: int
    plan_len_ovlp: int
    plan_n_diff_steps: int
    plan_use_padbuf: bool
    plan_extra_pad: int
    plan_max_path_length: int
    plan_max_ori_path_len: Optional[int]

    invdyn_horizon: int
    invdyn_max_path_length: int

    eval_probs_h5: str
    eval_default_n_comp: int
    eval_repl_thres: float
    eval_max_n_repl: int
    eval_ada_minus_n_wp: int
    eval_cond2_extra: int
    eval_n_max_steps: int

    @staticmethod
    def from_csv_row(row: Dict[str, str]) -> "EnvSpec":
        def parse_tuple(s: str) -> Tuple[int, ...]:
            # "(0,1)" -> (0,1)
            v = ast.literal_eval(s)
            return tuple(int(x) for x in v)

        return EnvSpec(
            env_name=row["env_name"],
            family=row["family"],
            scale=row["scale"],
            train_traj_len=int(row["train_traj_len"]),
            obs_dim_full=int(row["obs_dim_full"]),
            goal_dim=int(row["goal_dim"]),
            plan_obs_select_dim=parse_tuple(row["plan_obs_select_dim"]),
            plan_sm_horizon=int(row["plan_sm_horizon"]),
            plan_len_ovlp=int(row["plan_len_ovlp"]),
            plan_n_diff_steps=int(row["plan_n_diff_steps"]),
            plan_use_padbuf=(row["plan_use_padbuf"].strip().lower() in ["true", "1", "yes"]),
            plan_extra_pad=int(row["plan_extra_pad"]),
            plan_max_path_length=int(row["plan_max_path_length"]),
            plan_max_ori_path_len=(lambda v: None if v <= 0 else v)(int(row["plan_max_ori_path_len"])),
            invdyn_horizon=int(row["invdyn_horizon"]),
            invdyn_max_path_length=int(row["invdyn_max_path_length"]),
            eval_probs_h5=row["eval_probs_h5"],
            eval_default_n_comp=int(row["eval_default_n_comp"]),
            eval_repl_thres=float(row["eval_repl_thres"]),
            eval_max_n_repl=int(row["eval_max_n_repl"]),
            eval_ada_minus_n_wp=int(row["eval_ada_minus_n_wp"]),
            eval_cond2_extra=int(row["eval_cond2_extra"]),
            eval_n_max_steps=int(row["eval_n_max_steps"]),
        )

def load_env_specs(csv_path: str) -> Dict[str, EnvSpec]:
    """Load all env specs from ``csv_path`` keyed by environment name."""
    specs: Dict[str, EnvSpec] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spec = EnvSpec.from_csv_row(row)
            specs[spec.env_name] = spec
    if not specs:
        raise RuntimeError(f"No env specs loaded from {csv_path}")
    return specs


# Normalization classes

class LimitsNormalizer:
    """Min/max normalizer mapping values to/from the [-1, 1] range."""

    def __init__(self, mins: np.ndarray, maxs: np.ndarray):
        self.mins_np = np.asarray(mins, dtype=np.float32)
        self.maxs_np = np.asarray(maxs, dtype=np.float32)

    @property
    def mins(self):
        return self.mins_np

    @property
    def maxs(self):
        return self.maxs_np

    def normalize(self, x):
        if torch.is_tensor(x):
            mins = torch.as_tensor(self.mins_np, device=x.device, dtype=x.dtype)
            maxs = torch.as_tensor(self.maxs_np, device=x.device, dtype=x.dtype)
            y = (x - mins) / (maxs - mins)
            return 2.0 * y - 1.0
        x = np.asarray(x, dtype=np.float32)
        y = (x - self.mins_np) / (self.maxs_np - self.mins_np)
        return 2.0 * y - 1.0

    def unnormalize(self, x, eps: float = 1e-4, clip: bool = True):
        if torch.is_tensor(x):
            y = x
            if clip:
                y = torch.clamp(y, -1.0, 1.0)
            y = (y + 1.0) / 2.0
            mins = torch.as_tensor(self.mins_np, device=x.device, dtype=x.dtype)
            maxs = torch.as_tensor(self.maxs_np, device=x.device, dtype=x.dtype)
            return y * (maxs - mins) + mins

        y = np.asarray(x, dtype=np.float32)
        if y.max() > 1.0 + eps or y.min() < -1.0 - eps:
            y = np.clip(y, -1.0, 1.0)
        y = (y + 1.0) / 2.0
        return y * (self.maxs_np - self.mins_np) + self.mins_np


class DatasetNormalizer:
    """Holds separate observation and action normalizers for a dataset."""

    def __init__(self, obs_mins: np.ndarray, obs_maxs: np.ndarray, act_mins: np.ndarray, act_maxs: np.ndarray):
        self.normalizers = {
            "observations": LimitsNormalizer(obs_mins, obs_maxs),
            "actions": LimitsNormalizer(act_mins, act_maxs),
        }
        print(f"[DatasetNormalizer] obs mins: {obs_mins}, maxs: {obs_maxs}")
        print(f"[DatasetNormalizer] act mins: {act_mins}, maxs: {act_maxs}")
        self.observation_dim = int(obs_mins.shape[0])
        self.action_dim = int(act_mins.shape[0]) if act_mins is not None else 0

    def normalize(self, x, key: str):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key: str):
        if key == "actions":
            return self.normalizers[key].unnormalize(x, clip=False)
        return self.normalizers[key].unnormalize(x, clip=True)

    # Convenience helpers for callers that only deal with observations/actions.
    def normalize_obs(self, x):
        return self.normalize(x, "observations")

    def unnormalize_obs(self, x):
        return self.unnormalize(x, "observations")

    def normalize_act(self, x):
        return self.normalize(x, "actions")

    def unnormalize_act(self, x):
        return self.unnormalize(x, "actions")
