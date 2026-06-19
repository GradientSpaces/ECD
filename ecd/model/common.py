"""Shared neural-network building blocks (positional embeddings, helpers)."""

import math

import torch
import torch.nn as nn


def zero_module(module: nn.Module, do_zero: bool):
    """Optionally zero-initialize all parameters of ``module`` and return it."""
    if do_zero:
        for p in module.parameters():
            p.data.fill_(0)
    return module


def count_parameters(model: nn.Module) -> int:
    """Return the total number of parameters in ``model``."""
    return sum(p.numel() for p in model.parameters())


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding used to encode diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
