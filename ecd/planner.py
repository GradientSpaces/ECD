"""CompDiffuser planner: VP Gaussian diffusion over trajectory chunks with compositional samplers (CD and ECD)."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .dataset.ogbench import EnvSpec
from .model.unet1d import ConditionalUNet1D
from .model.dit1d import ConditionalDiT1D
from .ecd_prior import FittedGaussianMarkovPrior



def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype=torch.float32) -> torch.Tensor:
    """Return the cosine beta schedule of Nichol & Dhariwal for ``timesteps`` steps."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas, dtype=dtype)


def extract_2d(a_1d: torch.Tensor, t_2d: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Gather schedule coefficients ``a_1d`` at per-(batch, horizon) timesteps ``t_2d``, broadcast to ``x_shape``."""
    assert a_1d.ndim == 1 and t_2d.ndim == 2
    b, h = t_2d.shape
    out = a_1d[t_2d]  # (B,H)
    return out.reshape(b, h, *((1,) * (len(x_shape) - 2)))


def apply_conditioning(x: torch.Tensor, conditions: Dict[int, torch.Tensor], action_dim: int) -> torch.Tensor:
    """In-place inpaint the observation dims of ``x`` at each conditioned horizon index ``t``."""
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


def _repeat_batch_tensor(v: torch.Tensor, n_rp: int) -> torch.Tensor:
    """Tile a tensor ``n_rp`` times along the batch dimension."""
    return v.repeat((n_rp,) + (1,) * (v.ndim - 1))


def batch_repeat_tensor_in_dict(x: torch.Tensor, t_2d: torch.Tensor, d: Dict[str, Any], n_rp: int):
    """Batch-tile ``x``, ``t_2d`` and every batch-aligned tensor in ``d`` (used for classifier-free duplication)."""
    x2 = _repeat_batch_tensor(x, n_rp)
    t2 = _repeat_batch_tensor(t_2d, n_rp)
    d2 = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.shape[0] == x.shape[0]:
            d2[k] = _repeat_batch_tensor(v, n_rp)
        else:
            d2[k] = v
    return x2, t2, d2


class WeightedL2Loss(nn.Module):
    """Per-dimension weighted MSE loss used by the diffusion trainer."""

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor, ext_loss_w: float = 1.0):
        loss = ext_loss_w * F.mse_loss(pred, targ, reduction="none")
        return (loss * self.weights).mean(), {}


# Main class

class ChunkDiffusion(nn.Module):
    """VP Gaussian diffusion over trajectory chunks, with compositional CD and ECD samplers for multi-chunk plans."""

    def __init__(
        self,
        model: nn.Module,
        horizon: int,
        observation_dim: int,
        n_timesteps: int,
        clip_denoised: bool = True,
        predict_epsilon: bool = False,
        diff_config: Dict[str, Any] = None,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.observation_dim = int(observation_dim)
        self.model = model

        diff_config = diff_config or {}
        self.diff_config = diff_config
        self.len_ovlp_cd = int(diff_config["len_ovlp_cd"])

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = bool(clip_denoised)
        self.predict_epsilon = bool(predict_epsilon)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        loss_weights = torch.ones((self.horizon, self.observation_dim), dtype=torch.float32)
        self.loss_fn = WeightedL2Loss(loss_weights)

        self.infer_deno_type = str(diff_config.get("infer_deno_type", "same"))
        self.w_loss_type = str(diff_config.get("w_loss_type", "all"))
        self.tr_inpat_prob = float(diff_config["tr_inpat_prob"])
        self.tr_ovlp_prob = float(diff_config["tr_ovlp_prob"])
        self.tr_1side_drop_prob = float(diff_config["tr_1side_drop_prob"])
        assert abs(self.tr_inpat_prob + self.tr_ovlp_prob - 1.0) < 1e-6

        self.condition_guidance_w = float(diff_config.get("condition_guidance_w", 2.0))
        self.var_temp = 1.0

        # DDIM
        self.num_train_timesteps = self.n_timesteps
        self.ddim_num_inference_steps = int(diff_config.get("ddim_steps", 50))
        self.ddim_t_power = float(diff_config.get("ddim_t_power", 1.0))
        self.ddim_eta = 1.0
        self.use_ddim = True
        self.use_eta_noise = False
        self.final_alpha_cumprod = torch.tensor([1.0])

    def get_total_hzn(self, num_comp: int) -> int:
        """Length of the full stitched trajectory for ``num_comp`` overlapping chunks."""
        return int(num_comp * self.horizon - (num_comp - 1) * self.len_ovlp_cd)

    def predict_start_from_noise(self, x_t: torch.Tensor, t_2d: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Recover x0 from a noised sample; identity when the model predicts x0 directly."""
        if self.predict_epsilon:
            return extract_2d(self.sqrt_recip_alphas_cumprod, t_2d, x_t.shape) * x_t - \
                   extract_2d(self.sqrt_recipm1_alphas_cumprod, t_2d, x_t.shape) * noise
        return noise

    def predict_noise_from_start(self, x_t: torch.Tensor, t_2d: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Recover the epsilon noise implied by a (x_t, x0) pair."""
        return (extract_2d(self.sqrt_recip_alphas_cumprod, t_2d, x_t.shape) * x_t - x0) / \
               extract_2d(self.sqrt_recipm1_alphas_cumprod, t_2d, x_t.shape)

    # core

    def q_sample(self, x_start: torch.Tensor, t_2d: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: noise ``x_start`` to timestep ``t_2d`` (samples noise if not given)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        c1 = extract_2d(self.sqrt_alphas_cumprod, t_2d, x_start.shape)
        c2 = extract_2d(self.sqrt_one_minus_alphas_cumprod, t_2d, x_start.shape)
        return c1 * x_start + c2 * noise

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t_2d: torch.Tensor):
        """Mean, variance and log-variance of the diffusion posterior q(x_{t-1} | x_t, x0)."""
        mean = extract_2d(self.posterior_mean_coef1, t_2d, x_t.shape) * x_start + \
               extract_2d(self.posterior_mean_coef2, t_2d, x_t.shape) * x_t
        var = extract_2d(self.posterior_variance, t_2d, x_t.shape)
        log_var = extract_2d(self.posterior_log_variance_clipped, t_2d, x_t.shape)
        return mean, var, log_var

    def small_model_pred(self, x: torch.Tensor, t_2d: torch.Tensor, tj_cond: Dict[str, Any]) -> torch.Tensor:
        """Run the local chunk denoiser; assumes a single shared timestep across the horizon."""
        assert (t_2d[0] == t_2d[0, 0]).all()
        t_1d = t_2d[:, 0]
        return self.model(x, t_1d, tj_cond)

    def p_mean_variance(self, x: torch.Tensor, t_2d: torch.Tensor, tj_cond: Dict[str, Any], return_modelout: bool = False):
        """Reverse-step posterior stats (and optionally x0/eps) for a chunk, with classifier-free guidance when ``do_cond``."""
        if bool(tj_cond.get("do_cond", False)):
            x2, t2, tj2 = batch_repeat_tensor_in_dict(x, t_2d, tj_cond, n_rp=2)
            t1 = t2[:, 0]
            out = self.model(x2, t1, tj2, force_dropout=True, half_fd=True)
            out_cd = out[: len(x)]
            out_uc = out[len(x):]
            model_out = out_uc + self.condition_guidance_w * (out_cd - out_uc)
        else:
            model_out = self.small_model_pred(x, t_2d, tj_cond)

        x0 = self.predict_start_from_noise(x, t_2d, model_out)
        if self.clip_denoised:
            x0 = torch.clamp(x0, -1.0, 1.0)
        else:
            raise RuntimeError("clip_denoised must be True")

        mean, var, log_var = self.q_posterior(x0, x, t_2d)
        if return_modelout:
            pred_eps = self.predict_noise_from_start(x, t_2d, x0)
            return mean, var, log_var, x0, pred_eps
        return mean, var, log_var

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, tj_cond: Dict[str, Any], t_2d: torch.Tensor) -> torch.Tensor:
        """One ancestral (non-DDIM) reverse step; no noise is added at t==0."""
        b = x.shape[0]
        mean, _, log_var = self.p_mean_variance(x, t_2d, tj_cond)
        noise = self.var_temp * torch.randn_like(x)
        nonzero = (1.0 - (t_2d == 0).float()).reshape(b, self.horizon, *((1,) * (len(x.shape) - 2)))
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    # training conditioning

    def extract_ovlp_from_full(self, x: torch.Tensor):
        """Return detached copies of a chunk's leading and trailing overlap segments."""
        st = x[:, : self.len_ovlp_cd].detach().clone()
        ed = x[:, -self.len_ovlp_cd :].detach().clone()
        return st, ed

    def create_train_tj_cond(
        self,
        x_clean: torch.Tensor,
        x_noisy: torch.Tensor,
        t_1d_st: torch.Tensor,
        t_1d_end: torch.Tensor,
        cond_st_gl: Dict[int, torch.Tensor],
        is_rand: bool = True,
    ):
        """Build the training-time overlap/inpaint conditioning dict with random side dropout."""
        device = x_clean.device
        b = x_clean.shape[0]
        all_drop_prob = self.tr_1side_drop_prob

        st_is_all_drop = (torch.rand((b,), device=device) < all_drop_prob)
        end_is_all_drop = (torch.rand((b,), device=device) < all_drop_prob)

        st_cd_use_ovlp = (torch.rand((b,), device=device) < self.tr_ovlp_prob)
        end_cd_use_ovlp = (torch.rand((b,), device=device) < self.tr_ovlp_prob)

        st_cd_use_inpat = ~st_cd_use_ovlp
        end_cd_use_inpat = ~end_cd_use_ovlp

        st_cd_use_ovlp[st_is_all_drop] = False
        st_cd_use_inpat[st_is_all_drop] = False
        end_cd_use_ovlp[end_is_all_drop] = False
        end_cd_use_inpat[end_is_all_drop] = False

        if st_cd_use_inpat.any():
            x_noisy[st_cd_use_inpat] = apply_conditioning(
                x_noisy[st_cd_use_inpat], {0: cond_st_gl[0][st_cd_use_inpat]}, action_dim=0
            )

        if end_cd_use_inpat.any():
            x_noisy[end_cd_use_inpat] = apply_conditioning(
                x_noisy[end_cd_use_inpat], {self.horizon - 1: cond_st_gl[self.horizon - 1][end_cd_use_inpat]}, action_dim=0
            )

        if is_rand:
            t_1d_st = t_1d_st - torch.randint_like(t_1d_st, low=0, high=2)
            t_1d_end = t_1d_end - torch.randint_like(t_1d_end, low=0, high=2)

        t_1d_st = torch.clamp(t_1d_st, 0, self.n_timesteps - 1)
        t_1d_end = torch.clamp(t_1d_end, 0, self.n_timesteps - 1)

        t2_st = torch.repeat_interleave(t_1d_st[:, None], repeats=self.len_ovlp_cd, dim=1)
        t2_end = torch.repeat_interleave(t_1d_end[:, None], repeats=self.len_ovlp_cd, dim=1)

        st_traj = self.q_sample(x_clean[:, : self.len_ovlp_cd].detach().clone(), t2_st)
        end_traj = self.q_sample(x_clean[:, -self.len_ovlp_cd :].detach().clone(), t2_end)

        tj_cond = {
            "st_ovlp_is_drop": ~st_cd_use_ovlp,
            "end_ovlp_is_drop": ~end_cd_use_ovlp,
            "st_ovlp_traj": st_traj,
            "end_ovlp_traj": end_traj,
            "st_ovlp_t": t_1d_st,
            "end_ovlp_t": t_1d_end,
            "is_st_inpat": st_cd_use_inpat,
            "is_end_inpat": end_cd_use_inpat,
        }
        return x_noisy, tj_cond

    def p_losses(self, x_clean: torch.Tensor, x_noisy: torch.Tensor, noise: torch.Tensor, t_2d: torch.Tensor, tj_cond: Dict[str, Any]):
        """Denoising loss for one batch; inpainted boundary steps are excluded from the target."""
        loss_w = torch.ones_like(x_clean[:, :, :1])
        loss_w[tj_cond["is_st_inpat"], 0] = 0.0
        loss_w[tj_cond["is_end_inpat"], self.horizon - 1] = 0.0

        pred = self.small_model_pred(x_noisy, t_2d, tj_cond)
        if self.predict_epsilon:
            return self.loss_fn(pred, noise, ext_loss_w=loss_w)
        return self.loss_fn(pred, x_clean, ext_loss_w=loss_w)

    def loss(self, x_clean: torch.Tensor, cond_st_gl: Dict[int, torch.Tensor]):
        """Compute the diffusion training loss for a batch of clean trajectory chunks."""
        b = x_clean.shape[0]
        t_1d = torch.randint(0, self.n_timesteps, (b, 1), device=x_clean.device).long()
        t_2d = torch.repeat_interleave(t_1d, repeats=self.horizon, dim=1)

        noise = torch.randn_like(x_clean)
        x_noisy = self.q_sample(x_clean, t_2d, noise=noise)

        x_noisy, tj_cond = self.create_train_tj_cond(
            x_clean=x_clean, x_noisy=x_noisy,
            t_1d_st=t_1d[:, 0], t_1d_end=t_1d[:, 0].clone(),
            cond_st_gl=cond_st_gl, is_rand=True,
        )
        return self.p_losses(x_clean, x_noisy, noise, t_2d, tj_cond)

    # eval conditioning

    def create_eval_tj_cond(
        self,
        x_et: torch.Tensor,
        st_traj: Optional[torch.Tensor],
        end_traj: Optional[torch.Tensor],
        t_1d_st: torch.Tensor,
        t_1d_end: torch.Tensor,
        t_type: str,
        is_noisy: bool,
        stgl_cond: Dict[int, torch.Tensor],
    ):
        """Build the eval-time conditioning dict: inpaint start/goal where given, else carry overlap trajectories."""
        b = x_et.shape[0]
        device = x_et.device

        st_is_drop = None if st_traj is None else torch.zeros((b,), dtype=torch.bool, device=device)
        end_is_drop = None if end_traj is None else torch.zeros((b,), dtype=torch.bool, device=device)

        if 0 in stgl_cond:
            assert st_traj is None
            x_et = apply_conditioning(x_et, {0: stgl_cond[0]}, action_dim=0)
            is_st_inpat = torch.ones((b,), dtype=torch.bool, device=device)
        else:
            is_st_inpat = torch.zeros((b,), dtype=torch.bool, device=device)

        hm1 = self.horizon - 1
        if hm1 in stgl_cond:
            assert end_traj is None
            x_et = apply_conditioning(x_et, {hm1: stgl_cond[hm1]}, action_dim=0)
            is_end_inpat = torch.ones((b,), dtype=torch.bool, device=device)
        else:
            is_end_inpat = torch.zeros((b,), dtype=torch.bool, device=device)

        assert ((st_traj is not None) or (0 in stgl_cond)) and ((end_traj is not None) or (hm1 in stgl_cond))

        if t_type == "rand":
            t_1d_st = t_1d_st - torch.randint_like(t_1d_st, low=0, high=2)
            t_1d_end = t_1d_end - torch.randint_like(t_1d_end, low=0, high=2)
        elif t_type == "-1":
            t_1d_st = t_1d_st - torch.ones_like(t_1d_st)
            t_1d_end = t_1d_end - torch.ones_like(t_1d_end)
        elif t_type == "0":
            pass
        else:
            raise NotImplementedError

        t_1d_st = torch.clamp(t_1d_st, 0, self.n_timesteps - 1)
        t_1d_end = torch.clamp(t_1d_end, 0, self.n_timesteps - 1)

        if st_traj is not None and (not is_noisy):
            t2 = torch.repeat_interleave(t_1d_st[:, None], repeats=self.len_ovlp_cd, dim=1)
            st_traj = self.q_sample(st_traj, t2)
        elif st_traj is not None:
            st_traj = st_traj.clone()

        if end_traj is not None and (not is_noisy):
            t2 = torch.repeat_interleave(t_1d_end[:, None], repeats=self.len_ovlp_cd, dim=1)
            end_traj = self.q_sample(end_traj, t2)
        elif end_traj is not None:
            end_traj = end_traj.clone()

        tj_cond = {
            "st_ovlp_is_drop": st_is_drop,
            "end_ovlp_is_drop": end_is_drop,
            "st_ovlp_traj": st_traj,
            "end_ovlp_traj": end_traj,
            "st_ovlp_t": t_1d_st,
            "end_ovlp_t": t_1d_end,
            "is_st_inpat": is_st_inpat,
            "is_end_inpat": is_end_inpat,
        }
        return x_et, tj_cond

    # DDIM

    def ddim_set_timesteps(self, num_inference_steps: int) -> np.ndarray:
        """Return decreasing train-time indices used by all samplers.

        The default matches the released CD implementation: a uniform DDIM grid.
        Setting ``self.ddim_t_power > 1`` uses a shared power grid that allocates
        more finite DDIM steps near low-noise times; this is intentionally sampler
        shared, not ECD-specific.
        """
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 1:
            return np.array([self.num_train_timesteps - 1], dtype=np.int64)

        p = float(getattr(self, "ddim_t_power", 1.0))
        if abs(p - 1.0) < 1e-8:
            step_ratio = self.num_train_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            return timesteps

        frac = np.linspace(0.0, 1.0, num_inference_steps, dtype=np.float64)
        timesteps = np.rint((self.num_train_timesteps - 1) * (1.0 - frac) ** p).astype(np.int64)
        timesteps[0] = self.num_train_timesteps - 1
        timesteps[-1] = 0
        # Remove duplicates while preserving the decreasing order. Duplicates may
        # occur for aggressive powers near t=0.
        keep = [0]
        for i in range(1, len(timesteps)):
            if timesteps[i] != timesteps[keep[-1]]:
                keep.append(i)
        return timesteps[keep].copy()

    @torch.no_grad()
    def ddim_p_sample(self, x: torch.Tensor, tj_cond: Dict[str, Any], timesteps: torch.Tensor, eta: float, use_clipped_model_output: bool = True, return_x0: bool = False):
        """One DDIM reverse step for a single chunk; optionally returns the predicted x0."""
        prev_timestep = timesteps - (self.num_train_timesteps // self.ddim_num_inference_steps)

        alpha_t = extract_2d(self.alphas_cumprod, timesteps, x.shape)
        if prev_timestep[0, 0] >= 0:
            alpha_prev = extract_2d(self.alphas_cumprod, prev_timestep, x.shape)
        else:
            alpha_prev = extract_2d(self.final_alpha_cumprod.to(timesteps.device), torch.zeros_like(timesteps), x.shape)

        beta_t = 1.0 - alpha_t
        _, _, _, x0, pred_eps = self.p_mean_variance(x, timesteps, tj_cond, return_modelout=True)
        pred_original_sample = x0
        model_output = pred_eps

        variance = (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        std_dev_t = eta * torch.sqrt(torch.clamp(variance, min=0.0))

        if use_clipped_model_output:
            model_output = (x - torch.sqrt(alpha_t) * pred_original_sample) / torch.sqrt(torch.clamp(beta_t, min=1e-12))

        pred_dir = torch.sqrt(torch.clamp(1 - alpha_prev - std_dev_t**2, min=0.0)) * model_output
        prev_sample = torch.sqrt(alpha_prev) * pred_original_sample + pred_dir

        if self.use_eta_noise and eta > 0:
            prev_sample = prev_sample + std_dev_t * torch.randn_like(model_output)

        if return_x0:
            return prev_sample, pred_original_sample

        return prev_sample

    # non-composed conditional sample (n_comp==1)

    @torch.no_grad()
    def conditional_sample(self, stgl_cond: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Sample a single (n_comp==1) start/goal-conditioned trajectory via the reverse diffusion loop."""
        device = self.betas.device
        b = int(next(iter(stgl_cond.values())).shape[0])
        x = self.var_temp * torch.randn((b, self.horizon, self.observation_dim), device=device)

        time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps) if self.use_ddim else list(reversed(range(self.n_timesteps)))
        for ti in time_idx:
            t = torch.full((b, self.horizon), int(ti), device=device, dtype=torch.long)
            x, tj = self.create_eval_tj_cond(
                x_et=x, st_traj=None, end_traj=None,
                t_1d_st=t[:, 0], t_1d_end=t[:, 0],
                t_type="0", is_noisy=False, stgl_cond=stgl_cond,
            )
            tj["do_cond"] = True
            if self.use_ddim:
                x = self.ddim_p_sample(x, tj, t, eta=self.ddim_eta, use_clipped_model_output=True)
            else:
                x = self.p_sample(x, tj, t)

        x = apply_conditioning(x, stgl_cond, action_dim=0)
        return x

    # composed sampling variants

    @torch.no_grad()
    def comp_pred_p_loop_n(self, shape: Tuple[int, int, int], stgl_cond: Dict[int, torch.Tensor], n_comp: int):
        """CD interleave sampler: denoise n_comp chunks per step, sharing overlaps between neighbors."""
        b, hzn, d = shape
        device = self.betas.device
        x_list = [torch.randn(shape, device=device) for _ in range(n_comp)]
        time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps) if self.use_ddim else list(reversed(range(self.n_timesteps)))

        for ti in time_idx:
            t = torch.full((b, hzn), int(ti), device=device, dtype=torch.long)

            for i in range(n_comp):
                x_i = x_list[i]
                if i == 0:
                    x_ip1 = x_list[i + 1]
                    st_traj_2, _ = self.extract_ovlp_from_full(x_ip1)
                    x_i, tj = self.create_eval_tj_cond(
                        x_et=x_i, st_traj=None, end_traj=st_traj_2,
                        t_1d_st=t[:, 0], t_1d_end=t[:, 0],
                        t_type="0", is_noisy=True, stgl_cond={0: stgl_cond[0]},
                    )
                    tj["do_cond"] = True
                elif i < n_comp - 1:
                    x_im1 = x_list[i - 1]
                    _, end_traj_im1 = self.extract_ovlp_from_full(x_im1)
                    x_ip1 = x_list[i + 1]
                    st_traj_ip1, _ = self.extract_ovlp_from_full(x_ip1)
                    x_i, tj = self.create_eval_tj_cond(
                        x_et=x_i, st_traj=end_traj_im1, end_traj=st_traj_ip1,
                        t_1d_st=t[:, 0] - 1, t_1d_end=t[:, 0],
                        t_type="0", is_noisy=True, stgl_cond={},
                    )
                    tj["do_cond"] = True
                else:
                    x_im1 = x_list[i - 1]
                    _, end_traj_im1 = self.extract_ovlp_from_full(x_im1)
                    x_i, tj = self.create_eval_tj_cond(
                        x_et=x_i, st_traj=end_traj_im1, end_traj=None,
                        t_1d_st=t[:, 0] - 1, t_1d_end=t[:, 0],
                        t_type="0", is_noisy=True, stgl_cond={hzn - 1: stgl_cond[hzn - 1]},
                    )
                    tj["do_cond"] = True

                if self.use_ddim:
                    x_i = self.ddim_p_sample(x_i, tj, t, eta=self.ddim_eta, use_clipped_model_output=True)
                else:
                    x_i = self.p_sample(x_i, tj, t)
                x_list[i] = x_i

        x_list[0] = apply_conditioning(x_list[0], {0: stgl_cond[0]}, action_dim=0)
        x_list[-1] = apply_conditioning(x_list[-1], {hzn - 1: stgl_cond[hzn - 1]}, action_dim=0)
        return x_list

    # ---------------------------------------------------------------------
    # CDGS baseline (compute-scaled GSC): gsc_resampling
    # ---------------------------------------------------------------------
    def avg_ovlp_chunk_gsc(self, x_p_list):
        """GSC consensus: average the overlap band shared by each pair of neighboring chunks."""
        assert self.horizon / self.len_ovlp_cd >= 2, "chunks would overlap by more than one band."
        x_p_list = [x_p.clone() for x_p in x_p_list]
        ov = self.len_ovlp_cd
        for i in range(1, len(x_p_list)):
            avg = (x_p_list[i - 1][:, -ov:] + x_p_list[i][:, :ov]) / 2
            x_p_list[i - 1][:, -ov:] = avg
            x_p_list[i][:, :ov] = avg
        return x_p_list

    def undo_step(self, x_p, t_low, t_high):
        """Re-noise a chunk from diffusion step ``t_low`` back up to ``t_high`` (the resampling jump)."""
        for beta in self.betas[t_low + 1: t_high + 1]:
            x_p = (1 - beta) ** 0.5 * x_p + beta ** 0.5 * torch.randn_like(x_p)
        return x_p

    @torch.no_grad()
    def comp_pred_p_loop_n_gsc_resampling(self, shape: Tuple[int, int, int],
                                          stgl_cond: Dict[int, torch.Tensor], n_comp: int, U: int = 4):
        """CDGS sampler: GSC overlap-averaging with ``U`` denoise->re-noise resampling rounds per reverse
        step. ``U`` is the compute-scaling knob (about U x the planning NFE of the CD interleave sampler)."""
        assert n_comp >= 2
        device = self.betas.device
        batch_size, hzn = shape[0], shape[1]
        x_p_list = [torch.randn(shape, device=device) for _ in range(n_comp)]
        assert len(stgl_cond[0]) == shape[0], f"{len(stgl_cond[0])} vs {shape[0]}"
        time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)
        for step_i, i_t in enumerate(time_idx):
            timesteps = torch.full((batch_size, self.horizon), i_t, device=device, dtype=torch.long)
            for u_t in range(U):
                x_p_list = self.avg_ovlp_chunk_gsc(x_p_list)
                x_p_next = [None for _ in range(n_comp)]
                for i in range(n_comp):
                    x_p_i = x_p_list[i]
                    if i == 0:                                          # first chunk: condition on start
                        st_traj_next, _ = self.extract_ovlp_from_full(x_p_list[i + 1])
                        x_p_i, tj = self.create_eval_tj_cond(
                            x_et=x_p_i, st_traj=None, end_traj=st_traj_next,
                            t_1d_st=timesteps[:, 0], t_1d_end=timesteps[:, 0],
                            t_type="0", is_noisy=True, stgl_cond={0: stgl_cond[0]})
                        tj["end_ovlp_is_drop"] = None
                    elif i < n_comp - 1:                                # interior chunk: unconditional inpaint
                        tj = dict(st_ovlp_is_drop=None, end_ovlp_is_drop=None,
                                  is_st_inpat=torch.zeros_like(x_p_i[:, 0, 0]).to(torch.bool),
                                  is_end_inpat=torch.zeros_like(x_p_i[:, 0, 0]).to(torch.bool))
                    else:                                               # last chunk: condition on goal
                        _, end_traj_prev = self.extract_ovlp_from_full(x_p_list[i - 1])
                        x_p_i, tj = self.create_eval_tj_cond(
                            x_et=x_p_i, st_traj=end_traj_prev, end_traj=None,
                            t_1d_st=timesteps[:, 0], t_1d_end=timesteps[:, 0],
                            t_type="0", is_noisy=True, stgl_cond={hzn - 1: stgl_cond[hzn - 1]})
                        tj["st_ovlp_is_drop"] = None
                    tj["do_cond"] = False
                    if self.use_ddim:
                        x_p_i = self.ddim_p_sample(x_p_i, tj, timesteps, self.ddim_eta, use_clipped_model_output=True)
                    else:
                        x_p_i = self.p_sample(x_p_i, tj, timesteps)
                    if step_i < len(time_idx) - 1 and u_t < U - 1:      # re-noise unless last step / last round
                        x_p_i = self.undo_step(x_p_i, time_idx[step_i + 1], time_idx[step_i])
                    x_p_next[i] = x_p_i
                x_p_list = x_p_next
        x_p_list[0] = apply_conditioning(x_p_list[0], {0: stgl_cond[0]}, action_dim=0)
        x_p_list[-1] = apply_conditioning(x_p_list[-1], {hzn - 1: stgl_cond[hzn - 1]}, action_dim=0)
        return x_p_list

    @staticmethod
    def _clip_vec(v: torch.Tensor, clip: float) -> torch.Tensor:
        """Clip a vector to max L2 norm ``clip`` along the last dim; no-op when ``clip <= 0``."""
        if clip is None or float(clip) <= 0.0:
            return v
        n = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return v * torch.clamp(float(clip) / n, max=1.0)

    @staticmethod
    def _parse_dim_weights_tensor(weight_spec, dim: int, device, dtype) -> Optional[torch.Tensor]:
        """Parse a per-state-dim weight spec (tensor, sequence, or 'xy=1,rest=0.1' string) into a (1,1,dim) tensor."""
        if weight_spec is None:
            return None
        if torch.is_tensor(weight_spec):
            weights = weight_spec.detach().to(device=device, dtype=dtype)
        elif isinstance(weight_spec, (list, tuple, np.ndarray)):
            weights = torch.as_tensor(weight_spec, device=device, dtype=dtype)
        else:
            text = str(weight_spec).strip().lower()
            if text in ["", "none", "uniform"]:
                return None
            if ("xy" in text) or ("rest" in text) or ("last2" in text) or ("ball" in text):
                xy_w = 1.0
                rest_w = 1.0
                last2_w = None
                for part in text.replace(";", ",").split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "=" in part:
                        k, v = part.split("=", 1)
                    elif ":" in part:
                        k, v = part.split(":", 1)
                    else:
                        raise ValueError(f"Bad dim weight token {part!r}; use xy=1,rest=0.1 or comma floats.")
                    k = k.strip()
                    val = float(v)
                    if k in ["xy", "first2", "first_2"]:
                        xy_w = val
                    elif k in ["rest", "other", "others", "nonxy", "non_xy"]:
                        rest_w = val
                    elif k in ["last2", "last_2", "ball", "ballxy", "ball_xy"]:
                        last2_w = val
                    else:
                        raise ValueError(f"Unknown dim weight key {k!r}; expected xy/rest/last2.")
                weights = torch.full((dim,), rest_w, device=device, dtype=dtype)
                weights[:min(2, dim)] = xy_w
                if last2_w is not None and dim >= 2:
                    weights[-2:] = last2_w
            else:
                weights = torch.as_tensor([float(x.strip()) for x in text.split(",") if x.strip()], device=device, dtype=dtype)
        if weights.numel() == 1:
            weights = weights.repeat(dim)
        if weights.numel() != dim:
            raise ValueError(f"Expected {dim} dim weights, got {weights.numel()}.")
        return weights.reshape(1, 1, dim)

    @staticmethod
    def _snr_gate_from_alpha_bar(alpha_bar: torch.Tensor, logsnr_start: float, logsnr_end: float) -> float:
        """Map log-SNR(alpha_bar) to a [0, 1] gate ramping linearly between ``logsnr_start`` and ``logsnr_end``."""
        alpha_bar = alpha_bar.clamp(1e-6, 1.0 - 1e-6)
        logsnr = torch.log(alpha_bar / (1.0 - alpha_bar)).item()
        g = (logsnr - float(logsnr_start)) / max(1e-8, float(logsnr_end) - float(logsnr_start))
        return float(np.clip(g, 0.0, 1.0))

    @staticmethod
    def _solve_tridiag_const(a: float, b: float, c: float, rhs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Thomas-algorithm solve of a constant-coefficient tridiagonal system along the horizon dim."""
        bsz, n, _ = rhs.shape
        if n <= 0:
            return rhs
        device, dtype = rhs.device, rhs.dtype
        a_t = torch.tensor(float(a), device=device, dtype=dtype)
        b_t = torch.tensor(float(b), device=device, dtype=dtype)
        c_t = torch.tensor(float(c), device=device, dtype=dtype)
        if n == 1:
            return rhs / (b_t + eps)

        cp = torch.empty((n - 1,), device=device, dtype=dtype)
        cp[0] = c_t / (b_t + eps)
        for j in range(1, n - 1):
            cp[j] = c_t / (b_t - a_t * cp[j - 1] + eps)

        dp = torch.empty_like(rhs)
        dp[:, 0] = rhs[:, 0] / (b_t + eps)
        for j in range(1, n):
            dp[:, j] = (rhs[:, j] - a_t * dp[:, j - 1]) / (b_t - a_t * cp[j - 1] + eps)

        out = torch.empty((bsz, n, rhs.shape[-1]), device=device, dtype=dtype)
        out[:, -1] = dp[:, -1]
        for j in range(n - 2, -1, -1):
            out[:, j] = dp[:, j] - cp[j] * out[:, j + 1]
        return out

    def _compute_chunk_jvp(
        self,
        resid: torch.Tensor,
        w_mask: torch.Tensor,
        len_ov: int,
        chunk_idx: int,
        n_comp: int,
        jvp_h_state: float = 0.1,
        jvp_damping: float = 0.1,
        jvp_h_pair: float = 1.0,
        apply_to_overlap: bool = True,
    ) -> torch.Tensor:
        """Markov-surrogate chunk JVP: damped tridiagonal smooth of the interior residual, sent as overlap boundary messages."""
        b_s, hzn, _ = resid.shape
        if len_ov <= 0 or n_comp <= 1 or hzn <= 0:
            return torch.zeros_like(resid)

        left_bd = 1 if chunk_idx == 0 else len_ov
        right_bd = 1 if chunk_idx == n_comp - 1 else len_ov
        int_start = int(left_bd)
        int_end = int(hzn - right_bd)
        if int_end <= int_start:
            return torch.zeros_like(resid)

        r_int = (w_mask * resid)[:, int_start:int_end, :]
        u = self._solve_tridiag_const(
            a=-float(jvp_h_state),
            b=1.0 + 2.0 * float(jvp_h_state) + float(jvp_damping),
            c=-float(jvp_h_state),
            rhs=r_int,
        )
        msg_left = float(jvp_h_pair) * u[:, 0, :]
        msg_right = float(jvp_h_pair) * u[:, -1, :]
        d_jvp = torch.zeros_like(resid)
        if chunk_idx > 0:
            if apply_to_overlap:
                d_jvp[:, :len_ov, :] += msg_left[:, None, :]
            else:
                d_jvp[:, len_ov - 1, :] += msg_left
        if chunk_idx < n_comp - 1:
            if apply_to_overlap:
                d_jvp[:, -len_ov:, :] += msg_right[:, None, :]
            else:
                d_jvp[:, hzn - len_ov, :] += msg_right
        return d_jvp

    @torch.no_grad()
    def comp_pred_p_loop_n_chunk_ecd(
        self,
        shape: Tuple[int, int, int],
        stgl_cond: Dict[int, torch.Tensor],
        n_comp: int,
        ecd_config: Optional[Dict[str, Any]] = None,
    ):
        """CD-interleave sampler with no-extra-NFE chunk-space bridge/JVP corrections."""
        b_s, hzn, o_dim = shape
        device = self.betas.device
        cfg = ecd_config or {}
        bridge_scale = float(cfg.get("base_scale", 1.0))
        jvp_scale = float(cfg.get("react_scale", 0.1))
        clip = float(cfg.get("react_clip", 1.0))
        chunk_react_type = str(cfg.get("chunk_react_type", "markov")).lower()
        markov_type = str(cfg.get("markov_type", "laplacian")).lower()
        markov_prior = cfg.get("markov_prior", None)
        prior_phi_weight = float(cfg.get("prior_phi_weight", 1.0))
        prior_psi_weight = float(cfg.get("prior_psi_weight", 0.25))
        prior_boundary_scale = float(cfg.get("prior_boundary_scale", 1.0))
        prior_ridge = cfg.get("prior_ridge", None)
        if prior_ridge is not None:
            prior_ridge = float(prior_ridge)
        base_dim_weights = self._parse_dim_weights_tensor(
            cfg.get("base_dim_weights", None),
            o_dim,
            device=device,
            dtype=torch.float32,
        )
        react_dim_weights = self._parse_dim_weights_tensor(
            cfg.get("react_dim_weights", None),
            o_dim,
            device=device,
            dtype=torch.float32,
        )
        markov_reaction = chunk_react_type in ["markov", "approx", "laplacian"]
        if chunk_react_type in ["none", "off", "zero"]:
            jvp_scale = 0.0
        elif not markov_reaction:
            raise ValueError(f"Unknown ECD chunk_react_type={chunk_react_type}")
        jvp_h_state = float(cfg.get("chunk_jvp_h_state", 0.1))
        jvp_h_pair = float(cfg.get("chunk_jvp_h_pair", 1.0))
        jvp_damping = float(cfg.get("chunk_jvp_damping", 0.1))
        len_ov = int(self.len_ovlp_cd)

        x_list = [torch.randn(shape, device=device) for _ in range(n_comp)]
        time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps) if self.use_ddim else list(reversed(range(self.n_timesteps)))
        step_stats = []

        for ti in time_idx:
            ti_int = int(ti)
            t = torch.full((b_s, hzn), ti_int, device=device, dtype=torch.long)
            alpha_bar_t = self.alphas_cumprod[ti_int]
            sqrt_alpha_t = self.sqrt_alphas_cumprod[ti_int]
            g_jvp = self._snr_gate_from_alpha_bar(alpha_bar_t, logsnr_start=-1.5, logsnr_end=5.0)
            bridge_norm_acc = 0.0
            jvp_norm_acc = 0.0

            for i in range(n_comp):
                x_t_cur = x_list[i]
                if i == 0:
                    x_ip1 = x_list[i + 1]
                    st_traj_2, _ = self.extract_ovlp_from_full(x_ip1)
                    x_t_cur, tj = self.create_eval_tj_cond(
                        x_et=x_t_cur, st_traj=None, end_traj=st_traj_2,
                        t_1d_st=t[:, 0], t_1d_end=t[:, 0],
                        t_type="0", is_noisy=True, stgl_cond={0: stgl_cond[0]},
                    )
                    tj["do_cond"] = True
                elif i < n_comp - 1:
                    x_im1 = x_list[i - 1]
                    _, end_traj_im1 = self.extract_ovlp_from_full(x_im1)
                    x_ip1 = x_list[i + 1]
                    st_traj_ip1, _ = self.extract_ovlp_from_full(x_ip1)
                    x_t_cur, tj = self.create_eval_tj_cond(
                        x_et=x_t_cur, st_traj=end_traj_im1, end_traj=st_traj_ip1,
                        t_1d_st=t[:, 0] - 1, t_1d_end=t[:, 0],
                        t_type="0", is_noisy=True, stgl_cond={},
                    )
                    tj["do_cond"] = True
                else:
                    x_im1 = x_list[i - 1]
                    _, end_traj_im1 = self.extract_ovlp_from_full(x_im1)
                    x_t_cur, tj = self.create_eval_tj_cond(
                        x_et=x_t_cur, st_traj=end_traj_im1, end_traj=None,
                        t_1d_st=t[:, 0] - 1, t_1d_end=t[:, 0],
                        t_type="0", is_noisy=True, stgl_cond={hzn - 1: stgl_cond[hzn - 1]},
                    )
                    tj["do_cond"] = True

                if self.use_ddim:
                    x_prev, x0_pred = self.ddim_p_sample(
                        x_t_cur, tj, t, eta=self.ddim_eta, use_clipped_model_output=True, return_x0=True
                    )
                else:
                    x_prev = self.p_sample(x_t_cur, tj, t)
                    x0_pred = x_prev

                mu_t = sqrt_alpha_t.reshape(1, 1, 1) * x0_pred.detach()
                resid = x_t_cur - mu_t
                w_mask = torch.ones_like(resid[..., :1])
                w_mask[:, 0] = 0.0
                w_mask[:, -1] = 0.0
                if n_comp > 1 and len_ov > 0:
                    if i > 0:
                        w_mask[:, :len_ov] *= 0.5
                    if i < n_comp - 1:
                        w_mask[:, -len_ov:] *= 0.5

                delta = torch.zeros_like(x_prev)
                if bridge_scale > 0.0:
                    d_bridge_full = -w_mask * resid
                    if base_dim_weights is not None:
                        d_bridge_full = d_bridge_full * base_dim_weights.to(
                            device=d_bridge_full.device, dtype=d_bridge_full.dtype
                        )
                    d_bridge = torch.zeros_like(d_bridge_full)
                    for obs_i in range(o_dim):
                        d_bridge[..., obs_i] = self._clip_vec(d_bridge_full[..., obs_i], clip)
                    delta = delta + bridge_scale * d_bridge
                    bridge_norm_acc += float(torch.linalg.vector_norm((bridge_scale * d_bridge).detach()).item())
                if jvp_scale > 0.0 and n_comp > 1 and len_ov > 0:
                    if markov_type in ["fitted", "fitted_gaussian", "gaussian"]:
                        if markov_prior is None:
                            raise RuntimeError(
                                "ecd_chunk fitted Gaussian reaction requested but no prior was loaded. "
                                "Pass --ecd_markov_type fitted_gaussian --ecd_prior_path <path>."
                            )
                        if not isinstance(markov_prior, FittedGaussianMarkovPrior):
                            raise TypeError(f"markov_prior must be FittedGaussianMarkovPrior, got {type(markov_prior)}")
                        u = markov_prior.solve(
                            (w_mask * resid).detach(),
                            alpha_bar=alpha_bar_t,
                            overlap=len_ov,
                            has_left_condition=(i > 0),
                            has_right_condition=(i < n_comp - 1),
                            phi_weight=prior_phi_weight,
                            psi_weight=prior_psi_weight,
                            ridge=prior_ridge,
                        )
                        d_jvp = torch.zeros_like(resid)
                        if i > 0:
                            d_jvp[:, :len_ov, :] += markov_prior.boundary_message(
                                u[:, :len_ov, :],
                                alpha_bar=alpha_bar_t,
                                side="left",
                                psi_weight=prior_psi_weight,
                                boundary_scale=prior_boundary_scale,
                                ridge=prior_ridge,
                            )
                        if i < n_comp - 1:
                            d_jvp[:, -len_ov:, :] += markov_prior.boundary_message(
                                u[:, -len_ov:, :],
                                alpha_bar=alpha_bar_t,
                                side="right",
                                psi_weight=prior_psi_weight,
                                boundary_scale=prior_boundary_scale,
                                ridge=prior_ridge,
                            )
                    else:
                        d_jvp = self._compute_chunk_jvp(
                            resid=resid,
                            w_mask=w_mask,
                            len_ov=len_ov,
                            chunk_idx=i,
                            n_comp=n_comp,
                            jvp_h_state=jvp_h_state,
                            jvp_damping=jvp_damping,
                            jvp_h_pair=jvp_h_pair,
                            apply_to_overlap=True,
                        )
                    if react_dim_weights is not None:
                        d_jvp = d_jvp * react_dim_weights.to(device=d_jvp.device, dtype=d_jvp.dtype)
                    d_jvp = self._clip_vec(d_jvp, clip)
                    delta = delta + g_jvp * jvp_scale * d_jvp
                    jvp_norm_acc += float(torch.linalg.vector_norm((g_jvp * jvp_scale * d_jvp).detach()).item())
                x_list[i] = x_prev.detach() + delta

            step_stats.append({"t": ti_int, "base_norm": bridge_norm_acc, "reaction_norm": jvp_norm_acc})

        x_list[0] = apply_conditioning(x_list[0], {0: stgl_cond[0]}, action_dim=0)
        x_list[-1] = apply_conditioning(x_list[-1], {hzn - 1: stgl_cond[hzn - 1]}, action_dim=0)
        self.last_ecd_step_stats = step_stats
        return x_list


def build_planner_diffusion_for_env(spec: EnvSpec) -> ChunkDiffusion:
    """Construct the CompDiffuser planner diffusion model configured for one environment spec."""
    # model-independent: sm_horizon / len_ovlp / T
    sm_h = spec.plan_sm_horizon
    len_ov = spec.plan_len_ovlp
    T = spec.plan_n_diff_steps
    obs_dim = len(spec.plan_obs_select_dim)

    # model-dependent: overlap encoder hyperparams
    use_dit_high_dim = obs_dim == 15
    if use_dit_high_dim:
        # Matches the released OGBench high-dimensional CD config: AntMaze-o15d
        # uses a DiT local denoiser with DiT overlap encoders.
        ovlp_o_dim = 384
        ovlp_model_config = dict(
            c_traj_hzn=len_ov,
            in_dim=obs_dim,
            out_dim=ovlp_o_dim,
            hidden_size=ovlp_o_dim,
            depth=8,
            num_heads=6,
            mlp_ratio=4.0,
            tjti_enc_config=dict(
                frame_stack=4,
                w_init_type="no",
            ),
        )
        network_config = dict(
            ovlp_model_type="dit_enc",
            st_ovlp_model_config=ovlp_model_config,
            end_ovlp_model_config=ovlp_model_config,
            inpaint_token_dim=48,
            inpaint_token_type="const",
            t_cond_type="add",
            frame_stack=4,
        )
        model = ConditionalDiT1D(
            horizon=sm_h,
            transition_dim=obs_dim,
            hidden_size=768,
            depth=16,
            num_heads=12,
            mlp_ratio=4.0,
            learn_sigma=False,
            network_config=network_config,
        )
    elif spec.family in ["antM", "humM", "pointM"]:
        ovlp_o_dim = 256
        if spec.family in ["antM", "pointM"]:
            ovlp_dim_mults = (1, 2, 3, 4)
            cnn_out_dim = 128
            final_mlp_dims = [1280, 512, ovlp_o_dim]
        elif spec.family == "humM":
            ovlp_dim_mults = (1, 2, 3, 4, 5)
            cnn_out_dim = 160
            final_mlp_dims = [1360, 512, ovlp_o_dim]
        else:
            raise NotImplementedError(spec.family)

        ovlp_model_config = dict(
            c_traj_hzn=len_ov,
            in_dim=obs_dim,
            base_dim=32,
            dim_mults=ovlp_dim_mults,
            time_dim=32,
            out_dim=ovlp_o_dim,
            tjti_enc_config=dict(
                t_seq_encoder_type="mlp",
                cnn_out_dim=cnn_out_dim,
                final_mlp_dims=final_mlp_dims,
                f_conv_ks=3,
            ),
        )
        network_config = dict(
            cat_t_w=True,
            resblock_ksize=5,
            st_ovlp_model_config=ovlp_model_config,
            end_ovlp_model_config=ovlp_model_config,
            ext_cond_dim=2 * ovlp_o_dim,
            energy_mode=False,
            time_mlp_config=3,
            inpaint_token_dim=32,
            inpaint_token_type="const",
        )
        model = ConditionalUNet1D(
            horizon=sm_h,
            transition_dim=obs_dim,
            base_dim=128,
            dim_mults=(1, 2, 4, 8),
            time_dim=96,
            network_config=network_config,
        )
    else:
        raise NotImplementedError(spec.family)

    diff_config = dict(
        infer_deno_type="same",
        w_loss_type="all",
        is_direct_train=True,
        obs_manual_loss_weights={},
        len_ovlp_cd=len_ov,
        tr_1side_drop_prob=0.20,
        tr_inpat_prob=0.5,
        tr_ovlp_prob=0.5,
        tr_no_ovlp_none=False,
        ddim_steps=50,  # default; can override at eval
    )

    diffusion = ChunkDiffusion(
        model=model,
        horizon=sm_h,
        observation_dim=obs_dim,
        n_timesteps=T,
        clip_denoised=True,
        predict_epsilon=False,
        diff_config=diff_config,
    )
    return diffusion
