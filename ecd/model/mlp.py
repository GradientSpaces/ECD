"""MLP backbones for the OGBench inverse-dynamics model."""

from typing import Any, Dict, List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ogb_get_default_init_unif_a(weight: torch.Tensor, scale: float = 1.0):
    """Return the (-a, a) bounds of OGBench's default fan-average uniform init."""
    fan_out, fan_in = weight.data.shape
    avg_fan = (fan_in + fan_out) / 2.0
    a = math.sqrt(scale / avg_fan)
    return -a, a


class MLPBackbone(nn.Module):
    """Configurable GELU MLP encoder matching the OGBench actor architecture."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: Optional[int], activate_final: bool, mlp_config: Dict[str, Any]):
        super().__init__()
        layer_dim = [input_dim] + list(hidden_dims) + ([] if output_dim is None else [output_dim])
        modules = []
        num_layers = len(layer_dim) - 1

        act_f = mlp_config.get("act_f", "gelu")
        assert act_f == "gelu"
        act_fn = nn.GELU

        use_dpout = bool(mlp_config.get("use_dpout", False))
        prob_dpout = float(mlp_config.get("prob_dpout", 0.0))
        n_dpout_until = int(mlp_config.get("n_dpout_until", 2))

        for i in range(num_layers):
            lin = nn.Linear(layer_dim[i], layer_dim[i + 1])
            nn.init.zeros_(lin.bias)
            modules.append(lin)
            modules.append(act_fn())
            if use_dpout and i < num_layers - n_dpout_until:
                modules.append(nn.Dropout(p=prob_dpout))

        if not activate_final:
            modules = modules[:-1]

        self.encoder = nn.Sequential(*modules)
        self.output_dim = layer_dim[-1] if output_dim is None else output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class InvDynMLP(nn.Module):
    """Inverse-dynamics MLP that predicts an action from (observation, goal)."""

    def __init__(self, input_dim: int, action_dim: int, obs_dim: int, act_net_config: Dict[str, Any], inv_m_config: Dict[str, Any]):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.is_out_dist = bool(inv_m_config.get("is_out_dist", False))
        assert self.is_out_dist is False

        hidden_dims = list(inv_m_config["hidden_dims"])
        self.actor_net = MLPBackbone(input_dim, hidden_dims, output_dim=None, activate_final=True, mlp_config=act_net_config)

        self.mean_net = nn.Linear(hidden_dims[-1], action_dim, bias=True)
        a, b = ogb_get_default_init_unif_a(self.mean_net.weight, scale=float(inv_m_config["final_fc_init_scale"]))
        nn.init.uniform_(self.mean_net.weight, a=a, b=b)
        nn.init.zeros_(self.mean_net.bias)

    def forward(self, observations: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observations, goals], dim=-1)
        feat = self.actor_net(x)
        return self.mean_net(feat)

    def loss(self, x_t: torch.Tensor, x_goal: torch.Tensor, a_t: torch.Tensor):
        pred = self.forward(x_t, x_goal)
        return F.mse_loss(pred, a_t), {}
