"""1D U-Net diffusion backbone for CompDiffuser."""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from .common import SinusoidalPosEmb, zero_module


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1dBlock_dd(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, mish: bool = True, n_groups: int = 8, conv_zero_init: bool = False):
        super().__init__()
        act_fn = nn.Mish() if mish else nn.SiLU()
        self.block = nn.Sequential(
            zero_module(nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2), conv_zero_init),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            act_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualTemporalBlock_dd(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        horizon: int,
        kernel_size: int = 5,
        mish: bool = True,
        conv_zero_init: bool = False,
        resblock_config: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()
        resblock_config = resblock_config or {}
        force_residual_conv = bool(resblock_config.get("force_residual_conv", False))
        time_mlp_config = int(resblock_config.get("time_mlp_config", 3))

        self.blocks = nn.ModuleList([
            Conv1dBlock_dd(inp_channels, out_channels, kernel_size, mish=mish, conv_zero_init=False),
            Conv1dBlock_dd(out_channels, out_channels, kernel_size, mish=mish, conv_zero_init=conv_zero_init),
        ])

        act_fn = nn.Mish() if mish else nn.SiLU()
        if time_mlp_config == 3:
            self.time_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                act_fn,
                nn.Linear(embed_dim * 2, out_channels),
                Rearrange("batch t -> batch t 1"),
            )
        else:
            self.time_mlp = nn.Sequential(
                act_fn,
                nn.Linear(embed_dim, out_channels),
                Rearrange("batch t -> batch t 1"),
            )

        if not force_residual_conv:
            self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        else:
            self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class Hi_ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, embed_dim: int, horizon: int, kernel_size: int = 5):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, out_channels),
            Rearrange("batch t -> batch t 1"),
        )
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TrajTimeEncoder(nn.Module):
    """Conv1d-and-MLP encoder mapping a noised overlap chunk to a conditioning vector."""

    def __init__(
        self,
        c_traj_hzn: int,
        in_dim: int,
        base_dim: int,
        dim_mults: Tuple[int, ...],
        time_dim: int,
        out_dim: int,
        tjti_enc_config: Dict[str, Any],
    ):
        super().__init__()
        self.c_traj_hzn = int(c_traj_hzn)
        self.out_dim = int(out_dim)

        dims = [in_dim, *[round(base_dim * m) for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.t_seq_encoder_type = tjti_enc_config["t_seq_encoder_type"]
        assert self.t_seq_encoder_type == "mlp"
        self.cnn_out_dim = int(tjti_enc_config["cnn_out_dim"])
        self.f_conv_ks = int(tjti_enc_config["f_conv_ks"])
        self.final_mlp_dims = list(tjti_enc_config["final_mlp_dims"])
        self.mid_conv_ks = int(tjti_enc_config.get("mid_conv_ks", 5))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        res_block_type = Hi_ResidualTemporalBlock
        self.downs = nn.ModuleList([])
        horizon = self.c_traj_hzn
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                res_block_type(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                res_block_type(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))
            if not is_last:
                horizon //= 2

        mid_dim = dims[-1]
        self.mid_block1 = res_block_type(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, kernel_size=self.mid_conv_ks)
        self.mid_block2 = res_block_type(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, kernel_size=self.mid_conv_ks)

        self.final_conv = nn.Sequential(
            Conv1dBlock(mid_dim, mid_dim, kernel_size=self.f_conv_ks),
            nn.Conv1d(mid_dim, self.cnn_out_dim, 1),
        )

        self.last_hzn = int(horizon)
        f_mlp_in_dim = self.last_hzn * self.cnn_out_dim
        self.final_mlp_dims = [f_mlp_in_dim, *self.final_mlp_dims]
        assert self.final_mlp_dims[-1] == self.out_dim

        layers = []
        for i in range(len(self.final_mlp_dims) - 1):
            din, dout = self.final_mlp_dims[i], self.final_mlp_dims[i + 1]
            layers.append(nn.Linear(din, dout))
            if i != (len(self.final_mlp_dims) - 2):
                layers.append(nn.Mish())
        self.final_mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b h t -> b t h")
        t_feat = self.time_mlp(time)

        for res1, res2, down in self.downs:
            x = res1(x, t_feat)
            x = res2(x, t_feat)
            x = down(x)

        x = self.mid_block1(x, t_feat)
        x = self.mid_block2(x, t_feat)

        x = self.final_conv(x)             # (B, cnn_out_dim, last_hzn)
        x = einops.rearrange(x, "b t h -> b h t")
        x_flat = torch.flatten(x, start_dim=1, end_dim=2)
        return self.final_mlp(x_flat)


class ConditionalUNet1D(nn.Module):
    """CompDiffuser 1D U-Net conditioned on start/goal and neighbor overlap chunks."""

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        base_dim: int,
        dim_mults: Tuple[int, ...],
        time_dim: int,
        network_config: Dict[str, Any],
    ):
        super().__init__()
        self.network_config = network_config
        self.input_t_type = "1d"

        dims = [transition_dim, *[base_dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.cat_t_w = bool(network_config["cat_t_w"])
        self.resblock_ksize = int(network_config.get("resblock_ksize", 5))
        self.use_downup_sample = bool(network_config.get("use_downup_sample", True))

        self.st_ovlp_model = TrajTimeEncoder(**network_config["st_ovlp_model_config"])
        self.end_ovlp_model = TrajTimeEncoder(**network_config["end_ovlp_model_config"])
        self.ext_cond_dim = int(network_config["ext_cond_dim"])

        self.inpaint_token_dim = int(network_config["inpaint_token_dim"])
        self.inpaint_token_type = str(network_config["inpaint_token_type"])
        assert self.inpaint_token_type == "const"

        self.register_buffer("st_use_inpaint_token", torch.full((1, self.inpaint_token_dim), 1.0, dtype=torch.float32))
        self.register_buffer("st_no_inpaint_token", torch.full((1, self.inpaint_token_dim), 0.0, dtype=torch.float32))
        self.register_buffer("end_use_inpaint_token", torch.full((1, self.inpaint_token_dim), 1.0, dtype=torch.float32))
        self.register_buffer("end_no_inpaint_token", torch.full((1, self.inpaint_token_dim), 0.0, dtype=torch.float32))

        wall_embed_dim = self.st_ovlp_model.out_dim + self.end_ovlp_model.out_dim
        assert wall_embed_dim == self.ext_cond_dim
        tot_cond_dim = time_dim + wall_embed_dim + 2 * self.inpaint_token_dim

        act_fn = nn.Mish()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            act_fn,
            nn.Linear(time_dim * 2, time_dim * 2),
            act_fn,
            nn.Linear(time_dim * 2, time_dim),
        )

        self.force_residual_conv = bool(network_config.get("force_residual_conv", False))
        self.time_mlp_config = int(network_config.get("time_mlp_config", 3))
        resblock_config = dict(force_residual_conv=self.force_residual_conv, time_mlp_config=self.time_mlp_config)

        res_block_type = ResidualTemporalBlock_dd

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        horizon_cur = int(horizon)
        self.down_times = int(network_config.get("down_times", 10**9))

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1) or ind >= self.down_times
            self.downs.append(nn.ModuleList([
                res_block_type(dim_in, dim_out, embed_dim=tot_cond_dim, horizon=horizon_cur,
                               kernel_size=self.resblock_ksize, mish=True, conv_zero_init=False, resblock_config=resblock_config),
                res_block_type(dim_out, dim_out, embed_dim=tot_cond_dim, horizon=horizon_cur,
                               kernel_size=self.resblock_ksize, mish=True, conv_zero_init=False, resblock_config=resblock_config),
                Downsample1d(dim_out) if (not is_last and self.use_downup_sample) else nn.Identity(),
            ]))
            if not is_last:
                horizon_cur //= 2

        mid_dim = dims[-1]
        self.mid_block1 = res_block_type(mid_dim, mid_dim, embed_dim=tot_cond_dim, horizon=horizon_cur,
                                         kernel_size=self.resblock_ksize, mish=True, conv_zero_init=False, resblock_config=resblock_config)
        self.mid_block2 = res_block_type(mid_dim, mid_dim, embed_dim=tot_cond_dim, horizon=horizon_cur,
                                         kernel_size=self.resblock_ksize, mish=True, conv_zero_init=False, resblock_config=resblock_config)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1) or ind < (num_resolutions - self.down_times - 1)
            self.ups.append(nn.ModuleList([
                res_block_type(dim_out * 2, dim_in, embed_dim=tot_cond_dim, horizon=horizon_cur,
                               kernel_size=self.resblock_ksize, mish=True, conv_zero_init=False, resblock_config=resblock_config),
                res_block_type(dim_in, dim_in, embed_dim=tot_cond_dim, horizon=horizon_cur,
                               kernel_size=self.resblock_ksize, mish=True, conv_zero_init=False, resblock_config=resblock_config),
                Upsample1d(dim_in) if (not is_last and self.use_downup_sample) else nn.Identity(),
            ]))
            if not is_last:
                horizon_cur *= 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(base_dim, base_dim, kernel_size=self.resblock_ksize),
            nn.Conv1d(base_dim, transition_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,          # (B,H,D)
        time: torch.Tensor,       # (B,)
        tj_cond: Dict[str, Any],
        force_dropout: bool = False,
        half_fd: bool = False,
    ) -> torch.Tensor:
        b = x.shape[0]
        is_st_inpat = tj_cond["is_st_inpat"].to(torch.bool)
        is_end_inpat = tj_cond["is_end_inpat"].to(torch.bool)

        x = einops.rearrange(x, "b h t -> b t h")
        t_feat = self.time_mlp(time)

        st_ovlp_is_drop = tj_cond["st_ovlp_is_drop"]
        end_ovlp_is_drop = tj_cond["end_ovlp_is_drop"]

        if st_ovlp_is_drop is not None:
            st_feat = self.st_ovlp_model(tj_cond["st_ovlp_traj"], time=tj_cond["st_ovlp_t"])
            st_feat[st_ovlp_is_drop] = 0.0
        else:
            st_feat = torch.zeros((b, self.st_ovlp_model.out_dim), device=x.device, dtype=x.dtype)

        if end_ovlp_is_drop is not None:
            end_feat = self.end_ovlp_model(tj_cond["end_ovlp_traj"], time=tj_cond["end_ovlp_t"])
            end_feat[end_ovlp_is_drop] = 0.0
        else:
            end_feat = torch.zeros((b, self.end_ovlp_model.out_dim), device=x.device, dtype=x.dtype)

        st_token = torch.zeros((b, self.inpaint_token_dim), device=x.device, dtype=x.dtype)
        end_token = torch.zeros((b, self.inpaint_token_dim), device=x.device, dtype=x.dtype)
        n_st = int(is_st_inpat.sum().item())
        n_end = int(is_end_inpat.sum().item())

        if n_st > 0:
            st_token[is_st_inpat] = self.st_use_inpaint_token.repeat(n_st, 1)
        if n_st < b:
            st_token[~is_st_inpat] = self.st_no_inpaint_token.repeat(b - n_st, 1)
        if n_end > 0:
            end_token[is_end_inpat] = self.end_use_inpaint_token.repeat(n_end, 1)
        if n_end < b:
            end_token[~is_end_inpat] = self.end_no_inpaint_token.repeat(b - n_end, 1)

        if force_dropout:
            assert (not self.training) and half_fd and (b % 2 == 0)
            st_feat[b // 2:] = 0.0
            end_feat[b // 2:] = 0.0

        t_feat = torch.cat([t_feat, st_feat, end_feat, st_token, end_token], dim=-1)

        h = []
        for res1, res2, down in self.downs:
            x = res1(x, t_feat)
            x = res2(x, t_feat)
            h.append(x)
            x = down(x)

        x = self.mid_block1(x, t_feat)
        x = self.mid_block2(x, t_feat)

        for res1, res2, up in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = res1(x, t_feat)
            x = res2(x, t_feat)
            x = up(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x
