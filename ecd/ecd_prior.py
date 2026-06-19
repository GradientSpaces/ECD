"""Fitted Gaussian Markov prior for approximate ECD.

The fitted prior is deliberately lightweight: it estimates clean-state and
adjacent-pair Gaussian statistics in the planner's normalized observation space.
At diffusion step t it analytically pushes those covariances through the VP
forward noising kernel and uses the resulting precision blocks as a
block-tridiagonal Markov surrogate for the overlap reaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class GaussianMarkovPriorMeta:
    env_name: str
    obs_select_dim: Tuple[int, ...]
    dim: int
    n_states: int
    n_pairs: int
    ridge: float
    shrinkage: float
    version: int = 1


class FittedGaussianMarkovPrior:
    """Gaussian phi/psi Markov prior used by ``ecd_approx``.

    Stored statistics are clean-time moments.  Runtime methods construct the
    noised precision at the requested alpha_bar:

        Cov_t = alpha_bar * Cov_clean + (1-alpha_bar) * I.

    The pair precision over [x_i, x_{i+1}] supplies the temporal Hessian blocks
    and the boundary mixed-Hessian blocks.  The state precision supplies the
    unary phi curvature.
    """

    def __init__(
        self,
        state_mean: torch.Tensor,
        state_cov: torch.Tensor,
        pair_mean: torch.Tensor,
        pair_cov: torch.Tensor,
        meta: GaussianMarkovPriorMeta,
    ):
        self.state_mean = state_mean.detach().float().cpu()
        self.state_cov = state_cov.detach().float().cpu()
        self.pair_mean = pair_mean.detach().float().cpu()
        self.pair_cov = pair_cov.detach().float().cpu()
        self.meta = meta
        self._cache: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @property
    def dim(self) -> int:
        return int(self.meta.dim)

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> "FittedGaussianMarkovPrior":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        version = int(payload.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported ECD prior version {version} in {path}")
        meta = GaussianMarkovPriorMeta(
            env_name=str(payload["env_name"]),
            obs_select_dim=tuple(int(x) for x in payload["obs_select_dim"]),
            dim=int(payload["dim"]),
            n_states=int(payload["n_states"]),
            n_pairs=int(payload["n_pairs"]),
            ridge=float(payload.get("ridge", 1e-4)),
            shrinkage=float(payload.get("shrinkage", 0.0)),
            version=version,
        )
        return cls(
            state_mean=payload["state_mean"],
            state_cov=payload["state_cov"],
            pair_mean=payload["pair_mean"],
            pair_cov=payload["pair_cov"],
            meta=meta,
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": int(self.meta.version),
                "env_name": self.meta.env_name,
                "obs_select_dim": tuple(int(x) for x in self.meta.obs_select_dim),
                "dim": int(self.meta.dim),
                "n_states": int(self.meta.n_states),
                "n_pairs": int(self.meta.n_pairs),
                "ridge": float(self.meta.ridge),
                "shrinkage": float(self.meta.shrinkage),
                "state_mean": self.state_mean.cpu(),
                "state_cov": self.state_cov.cpu(),
                "pair_mean": self.pair_mean.cpu(),
                "pair_cov": self.pair_cov.cpu(),
            },
            path,
        )

    @staticmethod
    def _to_scalar_alpha(alpha_bar: float | torch.Tensor) -> float:
        if torch.is_tensor(alpha_bar):
            return float(alpha_bar.detach().reshape(-1)[0].cpu().item())
        return float(alpha_bar)

    def _precision_blocks(
        self,
        alpha_bar: float | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        ridge: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return noised precision blocks (Lambda_phi, L00, L01, L11)."""
        alpha = max(0.0, min(1.0, self._to_scalar_alpha(alpha_bar)))
        key = (round(alpha, 8), str(device), str(dtype), float(ridge))
        if key in self._cache:
            return self._cache[key]

        D = self.dim
        state_cov = self.state_cov.to(device=device, dtype=dtype)
        pair_cov = self.pair_cov.to(device=device, dtype=dtype)
        eye_d = torch.eye(D, device=device, dtype=dtype)
        eye_2d = torch.eye(2 * D, device=device, dtype=dtype)

        state_cov_t = alpha * state_cov + (1.0 - alpha) * eye_d + float(ridge) * eye_d
        pair_cov_t = alpha * pair_cov + (1.0 - alpha) * eye_2d + float(ridge) * eye_2d

        lambda_phi = torch.linalg.inv(state_cov_t)
        lambda_pair = torch.linalg.inv(pair_cov_t)
        l00 = lambda_pair[:D, :D]
        l01 = lambda_pair[:D, D:]
        l11 = lambda_pair[D:, D:]

        out = (lambda_phi, l00, l01, l11)
        # Avoid unbounded memory if users sweep many fractional alphas/ridges.
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = out
        return out

    @staticmethod
    def _solve_block(block: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Solve ``block @ x = rhs`` for tiny state blocks without cuBLAS."""
        D = block.shape[0]
        if D == 1:
            return rhs / block[0, 0]
        if D == 2:
            a, b = block[0, 0], block[0, 1]
            c, d = block[1, 0], block[1, 1]
            det = a * d - b * c
            x0 = (d * rhs[0] - b * rhs[1]) / det
            x1 = (-c * rhs[0] + a * rhs[1]) / det
            return torch.stack([x0, x1], dim=0)
        return torch.linalg.solve(block, rhs)

    @staticmethod
    def _block_tridiag_solve(
        block_diag: torch.Tensor,
        block_upper: torch.Tensor,
        block_lower: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        """Solve a block-tridiagonal SPD-ish system.

        Args:
            block_diag: (H,D,D)
            block_upper: (H-1,D,D), block at row i, col i+1
            block_lower: (H-1,D,D), block at row i+1, col i
            rhs: (B,H,D)
        """
        B, H, D = rhs.shape
        if H == 1:
            return FittedGaussianMarkovPrior._solve_block(block_diag[0], rhs[:, 0].T).T.reshape(B, 1, D)

        c_prime = []
        d_prime = []

        d0 = block_diag[0]
        c_prime.append(FittedGaussianMarkovPrior._solve_block(d0, block_upper[0]))
        d_prime.append(FittedGaussianMarkovPrior._solve_block(d0, rhs[:, 0].T).T)

        for i in range(1, H):
            d_tilde = block_diag[i] - block_lower[i - 1] @ c_prime[i - 1]
            rhs_tilde = rhs[:, i] - d_prime[i - 1] @ block_lower[i - 1].T
            if i < H - 1:
                c_prime.append(FittedGaussianMarkovPrior._solve_block(d_tilde, block_upper[i]))
            d_prime.append(FittedGaussianMarkovPrior._solve_block(d_tilde, rhs_tilde.T).T)

        out = [None for _ in range(H)]
        out[-1] = d_prime[-1]
        for i in range(H - 2, -1, -1):
            out[i] = d_prime[i] - out[i + 1] @ c_prime[i].T
        return torch.stack(out, dim=1)

    def solve(
        self,
        rhs: torch.Tensor,
        alpha_bar: float | torch.Tensor,
        overlap: int,
        has_left_condition: bool,
        has_right_condition: bool,
        phi_weight: float = 1.0,
        psi_weight: float = 0.25,
        ridge: Optional[float] = None,
        diagonal_jitter: float = 1e-5,
    ) -> torch.Tensor:
        """Solve H u = rhs using fitted phi/psi precision blocks."""
        if rhs.ndim != 3:
            raise ValueError(f"rhs must have shape (B,H,D), got {tuple(rhs.shape)}")
        B, H, D = rhs.shape
        if D != self.dim:
            raise ValueError(f"ECD prior dim mismatch: prior dim={self.dim}, rhs dim={D}")
        if H <= 1:
            return rhs

        device, dtype = rhs.device, rhs.dtype
        ridge_value = self.meta.ridge if ridge is None else float(ridge)
        lambda_phi, l00, l01, l11 = self._precision_blocks(alpha_bar, device, dtype, ridge_value)

        phi_weight = float(phi_weight)
        psi_weight = float(psi_weight)
        overlap = max(0, min(int(overlap), H))

        eye = torch.eye(D, device=device, dtype=dtype)
        diag_base = float(diagonal_jitter) * eye + phi_weight * lambda_phi
        block_diag = diag_base.unsqueeze(0).repeat(H, 1, 1)
        block_upper = psi_weight * l01.unsqueeze(0).repeat(H - 1, 1, 1)
        block_lower = psi_weight * l01.T.unsqueeze(0).repeat(H - 1, 1, 1)

        # Internal adjacent-pair potentials E_psi(y_i, y_{i+1}).
        block_diag[:-1] = block_diag[:-1] + psi_weight * l00
        block_diag[1:] = block_diag[1:] + psi_weight * l11

        # Boundary overlap-condition potentials.  They make the solve consistent
        # with the G blocks used by boundary_message().
        if overlap > 0 and has_left_condition:
            block_diag[:overlap] = block_diag[:overlap] + psi_weight * l11
        if overlap > 0 and has_right_condition:
            block_diag[-overlap:] = block_diag[-overlap:] + psi_weight * l00

        return self._block_tridiag_solve(block_diag, block_upper, block_lower, rhs)

    def boundary_message(
        self,
        u_boundary: torch.Tensor,
        alpha_bar: float | torch.Tensor,
        side: str,
        psi_weight: float = 0.25,
        boundary_scale: float = 1.0,
        ridge: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute approximate overlap reaction -G^T u on a boundary segment."""
        if u_boundary.numel() == 0:
            return u_boundary
        device, dtype = u_boundary.device, u_boundary.dtype
        ridge_value = self.meta.ridge if ridge is None else float(ridge)
        _lambda_phi, _l00, l01, _l11 = self._precision_blocks(alpha_bar, device, dtype, ridge_value)
        side = str(side).lower()
        psi_weight = float(psi_weight)
        boundary_scale = float(boundary_scale)

        # Pair ordering convention:
        # left boundary:  E_psi(c, y) => G = d^2E/dy dc = Lambda_10 = Lambda_01^T
        # right boundary: E_psi(y, c) => G = d^2E/dy dc = Lambda_01
        if side == "left":
            g_block = l01.T
        elif side == "right":
            g_block = l01
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side}")
        return -boundary_scale * psi_weight * torch.einsum("bod,df->bof", u_boundary, g_block)


def fit_gaussian_markov_prior_from_arrays(
    states: torch.Tensor,
    pairs: torch.Tensor,
    env_name: str,
    obs_select_dim: Tuple[int, ...],
    ridge: float = 1e-4,
    shrinkage: float = 0.02,
) -> FittedGaussianMarkovPrior:
    """Fit state and adjacent-pair Gaussian statistics from normalized tensors."""
    if states.ndim != 2:
        raise ValueError(f"states must have shape (N,D), got {tuple(states.shape)}")
    if pairs.ndim != 3 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (M,2,D), got {tuple(pairs.shape)}")
    states = states.detach().float().cpu()
    pairs = pairs.detach().float().cpu()
    D = states.shape[-1]
    pair_flat = pairs.reshape(pairs.shape[0], 2 * D)

    def mean_cov(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[0]
        if n < 2:
            raise ValueError("Need at least two samples to fit Gaussian prior")
        mu = x.mean(dim=0)
        xc = x - mu
        cov = (xc.T @ xc) / float(max(n - 1, 1))
        if shrinkage > 0.0:
            diag_cov = torch.diag(torch.diag(cov))
            cov = (1.0 - float(shrinkage)) * cov + float(shrinkage) * diag_cov
        cov = cov + float(ridge) * torch.eye(cov.shape[0], dtype=cov.dtype)
        return mu, cov

    state_mean, state_cov = mean_cov(states)
    pair_mean, pair_cov = mean_cov(pair_flat)
    meta = GaussianMarkovPriorMeta(
        env_name=env_name,
        obs_select_dim=tuple(int(x) for x in obs_select_dim),
        dim=int(D),
        n_states=int(states.shape[0]),
        n_pairs=int(pairs.shape[0]),
        ridge=float(ridge),
        shrinkage=float(shrinkage),
    )
    return FittedGaussianMarkovPrior(state_mean, state_cov, pair_mean, pair_cov, meta)
