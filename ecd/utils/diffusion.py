"""Diffusion noise-schedule helpers."""

import numpy as np
import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype=torch.float32) -> torch.Tensor:
    """Return the cosine beta schedule of Nichol & Dhariwal for ``timesteps`` steps."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas, dtype=dtype)
