"""Filesystem, checkpoint, and serialization helpers."""

from typing import Any, Optional

import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str) -> None:
    mkdir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[save_json] {path}")


def get_lr(opt: torch.optim.Optimizer) -> float:
    return float(opt.param_groups[0]["lr"])


def env_name_short(env_name: str) -> str:
    name = env_name.replace("stitch", "").replace("v0", "")
    while "--" in name:
        name = name.replace("--", "-")
    return name.strip("-_")


def load_text(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.read().strip() or None
    except FileNotFoundError:
        return None


def save_text(path: str, text: str) -> None:
    mkdir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)


def find_latest_checkpoint(logdir: str) -> Optional[str]:
    ckpts = glob.glob(os.path.join(logdir, "state_*.pt"))
    if not ckpts:
        return None

    def step_of(p: str) -> int:
        base = os.path.basename(p)
        num = base.replace("state_", "").replace(".pt", "")
        try:
            return int(num)
        except ValueError:
            return -1

    ckpts.sort(key=step_of)
    return ckpts[-1]


def checkpoint_path(logdir: str, epoch: str) -> str:
    if epoch == "latest":
        p = find_latest_checkpoint(logdir)
        if p is None:
            raise FileNotFoundError(f"No checkpoints under {logdir}")
        return p
    return os.path.join(logdir, f"state_{int(epoch)}.pt")


def remove_old_checkpoints(logdir: str, keep_last: int = 1) -> None:
    ckpts = glob.glob(os.path.join(logdir, "state_*.pt"))
    if len(ckpts) <= keep_last:
        return

    def step_of(p: str) -> int:
        base = os.path.basename(p)
        num = base.replace("state_", "").replace(".pt", "")
        try:
            return int(num)
        except ValueError:
            return -1

    ckpts.sort(key=step_of)
    old_ckpts = ckpts[:-keep_last]
    for p in old_ckpts:
        os.remove(p)
        print(f"[remove_old_checkpoints] removed {p}")


def freeze_model(m: nn.Module) -> None:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


def to_device(x: Any, device: str):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x