"""
DiT-based 1D diffusion backbone for CompDiffuser.
"""
import math

import einops
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp

from .common import count_parameters


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cond_dim=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        cond_dim = hidden_size if cond_dim is None else cond_dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    out_channels should directly be the env transition dim, e.g., 29 for antMaze.
    """
    def __init__(self, hidden_size, out_channels, cond_dim=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        cond_dim = hidden_size if cond_dim is None else cond_dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class DiTTrajTimeEncoder(nn.Module):
    """
    DiT-based trajectory encoder for overlap chunks; returns cls-token embedding.
    """
    def __init__(
        self,
        c_traj_hzn,
        in_dim,
        out_dim,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        tjti_enc_config=None,
    ):
        super().__init__()
        tjti_enc_config = tjti_enc_config or {}

        self.frame_stack = tjti_enc_config.get("frame_stack", 1)

        self.c_traj_hzn = c_traj_hzn
        self.out_dim = out_dim
        self.transition_dim = in_dim * self.frame_stack
        self.hidden_size = hidden_size

        self.out_channels = self.transition_dim
        self.num_heads = num_heads

        self.tjti_enc_config = tjti_enc_config

        # conditioning
        self.time_dim = hidden_size
        self.t_embedder = TimestepEmbedder(self.time_dim)

        # DiT backbone
        self.x_embedder = nn.Linear(in_features=self.transition_dim, out_features=hidden_size)

        self.num_patches = self.c_traj_hzn // self.frame_stack + 1
        assert self.c_traj_hzn % self.frame_stack == 0

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        self.cls_token = nn.Parameter(data=torch.randn(1, 1, hidden_size))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, cond_dim=hidden_size)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, out_channels=hidden_size, cond_dim=hidden_size)

        assert out_dim == hidden_size, "out_dim must match hidden_size"

        w_init_type = tjti_enc_config.get("w_init_type", "no")
        if w_init_type == "dit1d":
            self.initialize_weights()
        elif w_init_type != "no":
            raise NotImplementedError(f"Unknown w_init_type: {w_init_type}")

        print(f"[DiTTrajTimeEncoder] num_patches={self.num_patches}, hidden_size={hidden_size}, depth={depth}")
        print(f"[DiTTrajTimeEncoder] params={count_parameters(self)}")

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        assert (self.num_patches - 1) * self.frame_stack == self.c_traj_hzn
        tmp_pos_arr = np.arange(self.num_patches, dtype=np.int32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.hidden_size, pos=tmp_pos_arr)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.xavier_uniform_(self.x_embedder.weight.data)
        nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.cls_token.data, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, time):
        if self.frame_stack > 1:
            x = einops.rearrange(x, "b (t fs) dim -> b t (fs dim)", fs=self.frame_stack)

        x_input_emb = self.x_embedder(x)
        b_s, _, _ = x.shape
        cls_tokens = einops.repeat(self.cls_token, "1 1 d -> b 1 d", b=b_s)

        x_input_emb = torch.cat((cls_tokens, x_input_emb), dim=1)
        x_input_emb = x_input_emb + self.pos_embed

        t_feat = self.t_embedder(time)

        x = x_input_emb
        for block in self.blocks:
            x = block(x, t_feat)

        x = self.final_layer(x, t_feat)
        return x[:, 0, :]


class ConditionalDiT1D(nn.Module):
    """
    Diffusion model (CompDiffuser) with a Transformer backbone.
    """
    def __init__(
        self,
        horizon,
        transition_dim,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        network_config=None,
    ):
        super().__init__()
        network_config = network_config or {}
        self.learn_sigma = learn_sigma
        self.frame_stack = network_config.get("frame_stack", 1)

        self.horizon = horizon
        transition_dim = transition_dim * self.frame_stack
        self.transition_dim = transition_dim
        self.hidden_size = hidden_size

        self.out_channels = transition_dim * 2 if learn_sigma else transition_dim
        self.num_heads = num_heads
        self.network_config = network_config

        ovlp_model_type = network_config["ovlp_model_type"]
        self.st_ovlp_model_config = network_config["st_ovlp_model_config"]
        self.end_ovlp_model_config = network_config["end_ovlp_model_config"]

        if ovlp_model_type == "dit_enc":
            self.st_ovlp_model = DiTTrajTimeEncoder(**self.st_ovlp_model_config)
            self.end_ovlp_model = DiTTrajTimeEncoder(**self.end_ovlp_model_config)
        else:
            raise NotImplementedError(f"ovlp_model_type {ovlp_model_type}")

        self.create_inpat_nets()

        ovlp_2_cond_dim = self.st_ovlp_model.out_dim + self.end_ovlp_model.out_dim
        self.t_cond_type = network_config["t_cond_type"]

        if self.t_cond_type == "add":
            tot_cond_dim = ovlp_2_cond_dim + 2 * self.inpaint_token_dim
            self.time_dim = tot_cond_dim
            self.t_embedder = TimestepEmbedder(self.time_dim)
        else:
            raise NotImplementedError(self.t_cond_type)

        self.x_embedder = nn.Linear(in_features=transition_dim, out_features=hidden_size)

        self.num_patches = self.horizon // self.frame_stack
        assert self.horizon % self.frame_stack == 0

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, cond_dim=tot_cond_dim)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, self.out_channels, cond_dim=tot_cond_dim)

        self.initialize_weights()
        print(f"[ConditionalDiT1D] hidden_size={hidden_size} depth={depth} tot_cond_dim={tot_cond_dim}")
        self.input_t_type = "1d"

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        assert self.num_patches * self.frame_stack == self.horizon
        tmp_pos_arr = np.arange(self.num_patches, dtype=np.int32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.hidden_size, pos=tmp_pos_arr)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.xavier_uniform_(self.x_embedder.weight.data)
        nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def create_inpat_nets(self):
        self.st_inpaint_model = nn.Identity()
        self.end_inpaint_model = nn.Identity()
        self.inpaint_token_dim = self.network_config["inpaint_token_dim"]
        self.inpaint_token_type = self.network_config["inpaint_token_type"]
        if self.inpaint_token_type == "const":
            self.register_buffer(
                "st_use_inpaint_token",
                torch.full(size=(1, self.inpaint_token_dim), fill_value=1.0, dtype=torch.float32),
            )
            self.register_buffer(
                "st_no_inpaint_token",
                torch.full(size=(1, self.inpaint_token_dim), fill_value=0.0, dtype=torch.float32),
            )
            self.register_buffer(
                "end_use_inpaint_token",
                torch.full(size=(1, self.inpaint_token_dim), fill_value=1.0, dtype=torch.float32),
            )
            self.register_buffer(
                "end_no_inpaint_token",
                torch.full(size=(1, self.inpaint_token_dim), fill_value=0.0, dtype=torch.float32),
            )
        elif self.inpaint_token_type == "learn_if_inpt":
            self.st_use_inpaint_token = nn.Parameter(torch.zeros(1, self.inpaint_token_dim), requires_grad=True)
            nn.init.normal_(self.st_use_inpaint_token, std=0.02)
            self.register_buffer(
                "st_no_inpaint_token",
                torch.full(size=(1, self.inpaint_token_dim), fill_value=0.0, dtype=torch.float32),
            )
            self.end_use_inpaint_token = nn.Parameter(torch.zeros(1, self.inpaint_token_dim), requires_grad=True)
            nn.init.normal_(self.end_use_inpaint_token, std=0.02)
            self.register_buffer(
                "end_no_inpaint_token",
                torch.full(size=(1, self.inpaint_token_dim), fill_value=0.0, dtype=torch.float32),
            )
        else:
            raise NotImplementedError(self.inpaint_token_type)

    def forward(self, x, time, tj_cond: dict, force_dropout=False, half_fd=False):
        is_st_inpat = tj_cond["is_st_inpat"]
        is_end_inpat = tj_cond["is_end_inpat"]
        b_size = x.shape[0]
        assert is_st_inpat.shape[0] == b_size and is_st_inpat.ndim == 1 and is_st_inpat.dtype == torch.bool
        assert is_end_inpat.shape[0] == b_size and is_end_inpat.ndim == 1 and is_end_inpat.dtype == torch.bool

        if self.frame_stack > 1:
            x = einops.rearrange(x, "b (t fs) dim -> b t (fs dim)", fs=self.frame_stack)

        x_input_emb = self.x_embedder(x)
        x_input_emb = x_input_emb + self.pos_embed

        st_ovlp_is_drop = tj_cond["st_ovlp_is_drop"]
        end_ovlp_is_drop = tj_cond["end_ovlp_is_drop"]

        if st_ovlp_is_drop is not None:
            st_ovlp_feat = self.st_ovlp_model(tj_cond["st_ovlp_traj"], time=tj_cond["st_ovlp_t"])
            st_ovlp_feat[st_ovlp_is_drop] = 0.0
            assert not torch.logical_and(~st_ovlp_is_drop, is_st_inpat).any()
        else:
            st_ovlp_feat = torch.zeros((x.shape[0], self.st_ovlp_model.out_dim), device=x.device)

        if end_ovlp_is_drop is not None:
            end_ovlp_feat = self.end_ovlp_model(tj_cond["end_ovlp_traj"], time=tj_cond["end_ovlp_t"])
            end_ovlp_feat[end_ovlp_is_drop] = 0.0
            assert not torch.logical_and(~end_ovlp_is_drop, is_end_inpat).any()
        else:
            end_ovlp_feat = torch.zeros((x.shape[0], self.end_ovlp_model.out_dim), device=x.device)

        if self.inpaint_token_type == "const":
            st_token = torch.zeros(size=(b_size, self.inpaint_token_dim), dtype=x.dtype, device=x.device)
            num_st_inpt = torch.sum(is_st_inpat).item()
            st_token[is_st_inpat] = self.st_use_inpaint_token.repeat((num_st_inpt, 1))
            st_token[~is_st_inpat] = self.st_no_inpaint_token.repeat((b_size - num_st_inpt, 1))

            end_token = torch.zeros(size=(b_size, self.inpaint_token_dim), dtype=x.dtype, device=x.device)
            num_end_inpt = torch.sum(is_end_inpat).item()
            end_token[is_end_inpat] = self.end_use_inpaint_token.repeat((num_end_inpt, 1))
            end_token[~is_end_inpat] = self.end_no_inpaint_token.repeat((b_size - num_end_inpt, 1))

            st_token = self.st_inpaint_model(st_token)
            end_token = self.end_inpaint_model(end_token)
        elif self.inpaint_token_type == "learn_if_inpt":
            st_token = self.st_use_inpaint_token.repeat((b_size, 1))
            num_st_inpt = torch.sum(is_st_inpat).item()
            st_token[~is_st_inpat] *= 0.0
            st_token[~is_st_inpat] = self.st_no_inpaint_token.repeat((b_size - num_st_inpt, 1))

            end_token = self.end_use_inpaint_token.repeat((b_size, 1))
            num_end_inpt = torch.sum(is_end_inpat).item()
            end_token[~is_end_inpat] *= 0.0
            end_token[~is_end_inpat] = self.end_no_inpaint_token.repeat((b_size - num_end_inpt, 1))

            st_token = self.st_inpaint_model(st_token)
            end_token = self.end_inpaint_model(end_token)
        else:
            raise NotImplementedError(self.inpaint_token_type)

        if force_dropout:
            assert not self.training
            if half_fd:
                b_s = len(st_ovlp_feat)
                assert b_s % 2 == 0
                st_ovlp_feat[int(b_s // 2):] = 0.0
                end_ovlp_feat[int(b_s // 2):] = 0.0
            else:
                raise NotImplementedError("force_dropout without half_fd")

        if self.t_cond_type == "add":
            y_feat_cat = torch.cat([st_ovlp_feat, end_ovlp_feat, st_token, end_token], dim=-1)
            t_feat = self.t_embedder(time)
            c_feat_all = t_feat + y_feat_cat
        else:
            raise NotImplementedError(self.t_cond_type)

        x = x_input_emb

        for block in self.blocks:
            x = block(x, c_feat_all)
        x = self.final_layer(x, c_feat_all)

        if self.frame_stack > 1:
            x = einops.rearrange(x, "b t (fs dim) -> b (t fs) dim", fs=self.frame_stack)

        return x
