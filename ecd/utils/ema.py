"""Exponential moving average of model parameters."""

import torch
from torch import nn


class EMA:
    """Maintains an exponential moving average of another model's parameters."""

    def __init__(self, beta: float):
        self.beta = float(beta)

    @torch.no_grad()
    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module):
        """Update ``ma_model`` in place toward ``current_model`` with decay ``beta``."""
        for cur_p, ma_p in zip(current_model.parameters(), ma_model.parameters()):
            ma_p.data.mul_(self.beta).add_(cur_p.data, alpha=(1.0 - self.beta))
