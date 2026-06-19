"""Planning policy: sample candidate chunked plans, rank them, and blend the best."""

import time
from typing import Any, Dict, List, Optional, Tuple

import einops
import numpy as np
import torch

from .dataset.common import DatasetNormalizer
from .planner import ChunkDiffusion


# Utils: topK + blending

def _robust_rank_scale(scores: np.ndarray) -> np.ndarray:
    """Median/IQR-standardize scores so heterogeneous rankers can be summed on a common scale."""
    scores = np.asarray(scores, dtype=np.float64)
    med = float(np.median(scores))
    q25, q75 = np.percentile(scores, [25, 75])
    scale = max(float(q75 - q25), float(np.std(scores)), 1e-8)
    return (scores - med) / scale


def _parse_dim_weights(weight_spec: Optional[Any], dim: int) -> Optional[np.ndarray]:
    """Parse overlap-rank dim weights (array, sequence, or 'xy=1,rest=0.1' string) into a (1,1,dim) array, or None if uniform."""
    if weight_spec is None:
        return None
    if isinstance(weight_spec, np.ndarray):
        weights = weight_spec.astype(np.float32)
    elif isinstance(weight_spec, (list, tuple)):
        weights = np.asarray(weight_spec, dtype=np.float32)
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
            weights = np.full((dim,), rest_w, dtype=np.float32)
            weights[: min(2, dim)] = float(xy_w)
            if last2_w is not None and dim >= 2:
                weights[-2:] = float(last2_w)
        else:
            vals = [float(x) for x in text.split(",") if x.strip()]
            if len(vals) == 1:
                weights = np.full((dim,), vals[0], dtype=np.float32)
            elif len(vals) == 2 and dim > 2:
                weights = np.full((dim,), vals[1], dtype=np.float32)
                weights[:2] = vals[0]
            elif len(vals) == dim:
                weights = np.asarray(vals, dtype=np.float32)
            else:
                raise ValueError(f"Expected 1, 2, or {dim} dim weights, got {len(vals)}.")
    if weights.shape != (dim,):
        raise ValueError(f"Dim weights shape must be ({dim},), got {weights.shape}.")
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= 0.0:
        raise ValueError("At least one dim weight must be > 0.")
    if np.allclose(weights, weights[0]):
        return None
    return weights.reshape(1, 1, dim)


def compute_ovlp_dist(trajs_list_un: List[np.ndarray], len_ovlp_cd: int, dim_weights: Optional[Any] = None):
    """Rank candidate chunk lists by mean squared mismatch in their overlap regions (the fair ``overlap`` ranker)."""
    n_comp = len(trajs_list_un)
    dist_all = []
    for i in range(n_comp - 1):
        tj1 = trajs_list_un[i]
        tj2 = trajs_list_un[i + 1]
        end1 = tj1[:, -len_ovlp_cd:]
        st2 = tj2[:, :len_ovlp_cd]
        sq = (end1 - st2) ** 2
        weights = _parse_dim_weights(dim_weights, int(sq.shape[-1]))
        if weights is None:
            mse = sq.mean(axis=(1, 2))
        else:
            mse = (sq * weights).sum(axis=(1, 2)) / (float(len_ovlp_cd) * float(weights.sum()))
        dist_all.append(mse)
    dist_all = np.stack(dist_all, axis=1)
    dist_per_sample = dist_all.sum(axis=1)
    s_idxs = np.argsort(dist_per_sample)
    return s_idxs, dist_per_sample


def pick_top_n_trajs(trajs_list: List[np.ndarray], s_idxs: np.ndarray, top_n: int):
    """Select the top-``n`` candidates (by sorted index order) from each chunk array."""
    return [tj[s_idxs[:top_n]] for tj in trajs_list]


def blend_2_np_trajs_23d(traj_1: np.ndarray, traj_2: np.ndarray, blend_type: str = "exp", beta: float = 2.0):
    """Exponentially crossfade two overlapping trajectory segments along the time axis."""
    if traj_1.ndim == 2:
        L = traj_1.shape[0]
        t = np.arange(L)
        def w(tt):
            exponent = -beta * (tt - 0) / max(1e-8, (L - 1))
            return (np.exp(exponent) - np.exp(-beta)) / (1 - np.exp(-beta))
        weights = w(t)[:, None]
        return weights * traj_1 + (1 - weights) * traj_2

    B, L, D = traj_1.shape
    t = np.arange(L)
    def w(tt):
        exponent = -beta * (tt - 0) / max(1e-8, (L - 1))
        return (np.exp(exponent) - np.exp(-beta)) / (1 - np.exp(-beta))
    weights = w(t)[None, :, None]
    return weights * traj_1 + (1 - weights) * traj_2


class TrajectoryBlender:
    """Stitches a list of overlapping chunk trajectories into one full plan via crossfading."""

    def __init__(self, diffusion: ChunkDiffusion, normalizer: DatasetNormalizer, blend_type: str = "exp", exp_beta: float = 2.0):
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.blend_type = blend_type
        self.exp_beta = float(exp_beta)

        self.len_ovlp = diffusion.len_ovlp_cd
        self.hzn = diffusion.horizon
        self.hzn_step = self.hzn - self.len_ovlp
        self.gap_len = self.hzn - 2 * self.len_ovlp
        assert self.gap_len > 0

    def blend_traj_lists(self, trajs_list: List[np.ndarray], do_unnorm: bool) -> np.ndarray:
        """Blend a list of chunk trajectories into a single full-horizon plan array."""
        trajs_np = []
        for tj in trajs_list:
            if torch.is_tensor(tj):
                tj = tj.detach().cpu().numpy()
            trajs_np.append(tj.astype(np.float32))
        if do_unnorm:
            trajs_np = [self.normalizer.unnormalize(tj, "observations") for tj in trajs_np]

        n_comp = len(trajs_np)
        B, H, D = trajs_np[0].shape
        tot_hzn = self.diffusion.get_total_hzn(n_comp)
        out = np.zeros((B, tot_hzn, D), dtype=np.float32)

        for i in range(n_comp):
            tj = trajs_np[i]
            if i == 0:
                out[:, 0:self.hzn_step] = tj[:, :self.hzn_step]
            elif i < n_comp - 1:
                idx1 = self.hzn + (i - 1) * self.hzn_step
                idx2 = idx1 + self.gap_len
                out[:, idx1:idx2] = tj[:, self.len_ovlp:self.len_ovlp + self.gap_len]
            else:
                idx1 = self.hzn + (i - 1) * self.hzn_step
                idx2 = idx1 + self.hzn_step
                out[:, idx1:idx2] = tj[:, self.len_ovlp:]

        for i in range(n_comp - 1):
            idx1 = (i + 1) * self.hzn_step
            idx2 = idx1 + self.len_ovlp
            end_i = trajs_np[i][:, -self.len_ovlp:]
            st_ip1 = trajs_np[i + 1][:, :self.len_ovlp]
            out[:, idx1:idx2] = blend_2_np_trajs_23d(end_i, st_ip1, blend_type=self.blend_type, beta=self.exp_beta)

        return out


# Policy class

class CompositionalPolicy:
    """Start/goal-conditioned planning policy: samples candidate chunked plans, ranks them, and blends the best."""

    def __init__(
        self,
        diffusion_model: ChunkDiffusion,
        normalizer: DatasetNormalizer,
        ev_n_comp: int,
        ev_top_n: int,
        ev_pick_type: str,
        ev_cp_infer_t_type: str,
        tj_blend_type: str,
        tj_exp_beta: float,
        ecd_config: Optional[Dict[str, Any]] = None,
        rank_context: Optional[Dict[str, Any]] = None,
    ):
        self.diffusion_model = diffusion_model
        self.diffusion_model.eval()
        self.normalizer = normalizer

        self.n_comp = int(ev_n_comp)
        self.top_n = int(ev_top_n)
        assert ev_pick_type in ["first", "rand"]
        self.pick_type = ev_pick_type
        self.cp_infer_t_type = ev_cp_infer_t_type
        self.tj_blder = TrajectoryBlender(diffusion_model, normalizer, blend_type=tj_blend_type, exp_beta=tj_exp_beta)
        self.ecd_config = dict(ecd_config or {})
        self.rank_context = dict(rank_context or {})

        self.ncp_pred_time_list: List[Tuple[int, float]] = []
        self.ecd_rank_score_summary_list: List[Dict[str, float]] = []
        self.ecd_step_stats_list: List[List[Dict[str, float]]] = []

    @property
    def device(self):
        return next(self.diffusion_model.parameters()).device

    def _make_stgl_cond(self, st_xy: np.ndarray, gl_xy: np.ndarray, b_s: int):
        """Normalize start/goal observations and broadcast them into a batch-``b_s`` start/goal conditioning dict."""
        hzn = self.diffusion_model.horizon

        st_gl = np.stack([st_xy[None, :], gl_xy[None, :]], axis=0).astype(np.float32)  # (2,1,2)
        st_gl_nm = self.normalizer.normalize(st_gl, "observations")
        st_gl_nm = torch.as_tensor(st_gl_nm, device=self.device, dtype=torch.float32)
        return {
            0: einops.repeat(st_gl_nm[0], "n d -> (n r) d", r=b_s).clone(),
            hzn - 1: einops.repeat(st_gl_nm[1], "n d -> (n r) d", r=b_s).clone(),
        }

    def _sample_trajs_list(self, stgl_cond: Dict[int, torch.Tensor], b_s: int):
        """Dispatch to the configured compositional sampler (``cp_infer_t_type``) and return the candidate chunk lists."""
        hzn = self.diffusion_model.horizon
        o_dim = self.diffusion_model.observation_dim
        shape = (b_s, hzn, o_dim)
        t0 = time.time()
        if self.cp_infer_t_type == "interleave":
            trajs_list = self.diffusion_model.comp_pred_p_loop_n(shape, stgl_cond, n_comp=self.n_comp)
        elif self.cp_infer_t_type == "ecd_chunk":
            trajs_list = self.diffusion_model.comp_pred_p_loop_n_chunk_ecd(
                shape, stgl_cond, n_comp=self.n_comp, ecd_config=self.ecd_config
            )
        elif self.cp_infer_t_type == "gsc_resampling":
            trajs_list = self.diffusion_model.comp_pred_p_loop_n_gsc_resampling(
                shape, stgl_cond, n_comp=self.n_comp, U=int(self.ecd_config.get("gsc_u", 4))
            )
        else:
            raise NotImplementedError(f"cp_infer_t_type={self.cp_infer_t_type} is not implemented.")
        self.ncp_pred_time_list.append((self.n_comp, time.time() - t0))

        if self.cp_infer_t_type == "ecd_chunk":
            step_stats = getattr(self.diffusion_model, "last_ecd_step_stats", None)
            if step_stats:
                self.ecd_step_stats_list.append(step_stats)
        return trajs_list

    def _apply_distance_base_scale(self, st_xy: np.ndarray, gl_xy: np.ndarray) -> Optional[float]:
        """Optionally swap in an AntSoccer far-ball base scale; returns the previous value to restore afterward."""
        far_scale = self.ecd_config.get("base_far_scale", None)
        far_dist = self.ecd_config.get("base_far_dist", None)
        if far_scale is None or far_dist is None or len(st_xy) < 17 or len(gl_xy) < 17:
            return None

        prev = float(self.ecd_config.get("base_scale", 1.0))
        ball_dist = float(np.linalg.norm(np.asarray(st_xy[15:17]) - np.asarray(gl_xy[15:17])))
        self.ecd_config["base_scale"] = float(far_scale) if ball_dist >= float(far_dist) else prev
        return prev

    def _score_invdyn_action_candidates(self, candidates: np.ndarray) -> np.ndarray:
        """Lower is better. Scores candidate plans by predicted invdyn action demand."""
        inv_model = self.rank_context.get("inv_model", None)
        full_normalizer = self.rank_context.get("full_normalizer", None)
        current_full_obs = self.rank_context.get("current_full_obs", None)
        obs_select_dim = self.rank_context.get("obs_select_dim", None)
        if inv_model is None or full_normalizer is None or current_full_obs is None or obs_select_dim is None:
            raise ValueError("invdyn ranking requires inv_model, full_normalizer, current_full_obs, and obs_select_dim.")

        x = np.asarray(candidates, dtype=np.float32)
        if x.ndim != 3 or x.shape[1] < 2:
            return np.zeros((x.shape[0],), dtype=np.float32)
        obs_select_dim = tuple(int(i) for i in obs_select_dim)
        if len(obs_select_dim) != x.shape[-1]:
            raise ValueError(f"obs_select_dim length {len(obs_select_dim)} does not match candidate dim {x.shape[-1]}.")

        stride = max(1, int(self.ecd_config.get("rank_invdyn_stride", 4)))
        max_steps = int(self.ecd_config.get("rank_invdyn_max_steps", 96))
        batch_size = max(1, int(self.ecd_config.get("rank_invdyn_batch", 4096)))
        t_idxs = np.arange(0, x.shape[1] - 1, stride, dtype=np.int64)
        if max_steps > 0:
            t_idxs = t_idxs[:max_steps]
        if t_idxs.size == 0:
            return np.zeros((x.shape[0],), dtype=np.float32)

        bsz = x.shape[0]
        n_t = int(t_idxs.size)
        current_full_obs = np.asarray(current_full_obs, dtype=np.float32)
        full_obs = np.repeat(current_full_obs.reshape(1, -1), bsz * n_t, axis=0)
        full_obs[:, list(obs_select_dim)] = x[:, t_idxs, :].reshape(bsz * n_t, x.shape[-1])
        goals = x[:, t_idxs + 1, :].reshape(bsz * n_t, x.shape[-1])

        obs_nm = full_normalizer.normalize(full_obs, "observations")
        goal_nm = self.normalizer.normalize(goals, "observations")
        inv_device = next(inv_model.parameters()).device
        pred_chunks = []
        was_training = bool(inv_model.training)
        inv_model.eval()
        with torch.no_grad():
            for i in range(0, obs_nm.shape[0], batch_size):
                obs_t = torch.as_tensor(obs_nm[i:i + batch_size], device=inv_device, dtype=torch.float32)
                goal_t = torch.as_tensor(goal_nm[i:i + batch_size], device=inv_device, dtype=torch.float32)
                pred_chunks.append(inv_model(obs_t, goal_t).detach().cpu().numpy())
        if was_training:
            inv_model.train(True)

        acts = np.concatenate(pred_chunks, axis=0).reshape(bsz, n_t, -1)
        act_norm = np.linalg.norm(acts, axis=-1)
        act_abs = np.abs(acts)
        sat = np.maximum(act_abs - 0.85, 0.0).mean(axis=(1, 2))
        return (
            0.5 * act_norm[:, 0]
            + act_norm.mean(axis=1)
            + 0.5 * np.percentile(act_norm, 90, axis=1)
            + 2.0 * sat
        ).astype(np.float32)

    def _rank_and_blend(
        self,
        trajs_list: List[torch.Tensor],
        stgl_cond: Dict[int, torch.Tensor],
        gl_xy: np.ndarray,
        top_n: Optional[int] = None,
    ) -> np.ndarray:
        """Rank candidate chunk lists by the configured ranker and return one blended full plan."""
        is_ecd = self.cp_infer_t_type == "ecd_chunk"
        s_idxs = None
        rank_type = str(self.ecd_config.get("rank_type", "overlap")).lower() if is_ecd else ""

        trajs_np_un = [self.normalizer.unnormalize(tj.detach().cpu().numpy(), "observations") for tj in trajs_list]
        rank_dim_weights = self.ecd_config.get("rank_dim_weights", None)
        if is_ecd and rank_type in ["overlap_invdyn", "invdyn_overlap"]:
            blended_all = self.tj_blder.blend_traj_lists(trajs_np_un, do_unnorm=False)
            _, ov_scores = compute_ovlp_dist(trajs_np_un, self.diffusion_model.len_ovlp_cd, dim_weights=rank_dim_weights)
            invdyn_scores = self._score_invdyn_action_candidates(blended_all)
            scores = _robust_rank_scale(ov_scores) + _robust_rank_scale(invdyn_scores)
            s_idxs = np.argsort(scores)
            self.ecd_rank_score_summary_list.append(
                {
                    "min": float(np.min(scores)),
                    "mean": float(np.mean(scores)),
                    "max": float(np.max(scores)),
                }
            )
        elif is_ecd and rank_type != "overlap":
            raise ValueError(f"Unknown ECD rank_type={rank_type}")
        if s_idxs is None:
            s_idxs, _ = compute_ovlp_dist(trajs_np_un, self.diffusion_model.len_ovlp_cd, dim_weights=rank_dim_weights)
        topn = min(int(top_n if top_n is not None else self.top_n), len(s_idxs))
        topn_list = pick_top_n_trajs(trajs_np_un, s_idxs, topn)
        blended = self.tj_blder.blend_traj_lists(topn_list, do_unnorm=False)  # already unnorm
        if self.pick_type == "first":
            return blended[0]
        return blended[np.random.randint(0, topn)]

    @torch.no_grad()
    def plan(self, st_xy: np.ndarray, gl_xy: np.ndarray, b_s: int = 40) -> np.ndarray:
        """
        Returns: pick_traj in env xy (unnormalized) with length tot_hzn.
        """
        if self.n_comp == 1:
            stgl_cond = self._make_stgl_cond(st_xy, gl_xy, b_s)
            pred = self.diffusion_model.conditional_sample(stgl_cond)
            pred_un = self.normalizer.unnormalize(pred.detach().cpu().numpy(), "observations")
            return pred_un[0]

        stgl_cond = self._make_stgl_cond(st_xy, gl_xy, b_s)
        prev_base_scale = self._apply_distance_base_scale(st_xy, gl_xy)
        try:
            trajs_list = self._sample_trajs_list(stgl_cond, b_s)
        finally:
            if prev_base_scale is not None:
                self.ecd_config["base_scale"] = prev_base_scale
        return self._rank_and_blend(trajs_list, stgl_cond, gl_xy)

    @torch.no_grad()
    def plan_prefixes(self, st_xy: np.ndarray, gl_xy: np.ndarray, b_s: int, b_values: List[int]) -> Dict[int, np.ndarray]:
        """Plan once with batch ``b_s`` and return blended plans for each candidate-budget prefix in ``b_values``."""
        if self.n_comp == 1:
            plan = self.plan(st_xy, gl_xy, b_s=b_s)
            return {int(b): plan for b in b_values}

        b_values = sorted({max(1, min(int(b), int(b_s))) for b in b_values})
        stgl_cond = self._make_stgl_cond(st_xy, gl_xy, b_s)
        prev_base_scale = self._apply_distance_base_scale(st_xy, gl_xy)
        try:
            trajs_list = self._sample_trajs_list(stgl_cond, b_s)
        finally:
            if prev_base_scale is not None:
                self.ecd_config["base_scale"] = prev_base_scale
        out: Dict[int, np.ndarray] = {}
        for b in b_values:
            trajs_prefix = [tj[:b] for tj in trajs_list]
            stgl_prefix = {k: v[:b] for k, v in stgl_cond.items()}
            out[int(b)] = self._rank_and_blend(trajs_prefix, stgl_prefix, gl_xy, top_n=min(self.top_n, b))
        return out
