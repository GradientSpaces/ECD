"""Fit the Gaussian Markov prior used by approximate ECD.

Example:
    python -m src.compdiffuser.fit_ecd_prior \
      --env antmaze-giant-stitch-v0 \
      --out logs/antmaze-giant-stitch-v0/ecd_prior/gaussian_markov.pt
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch

from .dataset.common import load_env_specs
from .dataset.ogbench import make_normalizer, ogb_make_env, ogb_load_train_dataset, ogb_segment_episodes
from .ecd_prior import fit_gaussian_markov_prior_from_arrays


def _subsample(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if max_n is None or max_n <= 0 or x.shape[0] <= max_n:
        return x
    idx = rng.choice(x.shape[0], size=int(max_n), replace=False)
    return x[idx]


def fit_prior(args) -> str:
    """Fit and save the Gaussian Markov ECD prior for one environment; return the output path."""
    specs = load_env_specs(args.spec_csv)
    if args.env not in specs:
        raise ValueError(f"--env {args.env} not found in {args.spec_csv}. Available: {list(specs.keys())}")
    spec = specs[args.env]
    rng = np.random.default_rng(int(args.seed))

    normalizer = make_normalizer(spec.env_name, obs_select_dim=spec.plan_obs_select_dim)
    env = ogb_make_env(spec.env_name)
    dset = ogb_load_train_dataset(env)
    episodes = ogb_segment_episodes(dset["observations"], dset["actions"], dset["terminals"])
    try:
        env.close()
    except Exception:
        pass

    state_chunks: List[np.ndarray] = []
    pair_chunks: List[np.ndarray] = []
    dims = list(spec.plan_obs_select_dim)
    for obs_ep, _act_ep in episodes:
        if obs_ep.shape[0] < 2:
            continue
        obs_sel = obs_ep[:, dims].astype(np.float32)
        obs_nm = normalizer.normalize(obs_sel, "observations").astype(np.float32)
        state_chunks.append(obs_nm)
        pair_chunks.append(np.stack([obs_nm[:-1], obs_nm[1:]], axis=1))

    if not state_chunks or not pair_chunks:
        raise RuntimeError(f"No usable episodes found for {args.env}")
    states = np.concatenate(state_chunks, axis=0)
    pairs = np.concatenate(pair_chunks, axis=0)
    states = _subsample(states, int(args.max_states), rng)
    pairs = _subsample(pairs, int(args.max_pairs), rng)

    prior = fit_gaussian_markov_prior_from_arrays(
        states=torch.from_numpy(states),
        pairs=torch.from_numpy(pairs),
        env_name=spec.env_name,
        obs_select_dim=spec.plan_obs_select_dim,
        ridge=float(args.ridge),
        shrinkage=float(args.shrinkage),
    )

    out = args.out
    if out is None:
        out = os.path.join("logs", spec.env_name, "ecd_prior", "gaussian_markov.pt")
    prior.save(out)
    print(f"[ecd-prior] wrote {out}")
    print(f"[ecd-prior] env={spec.env_name} dim={prior.dim} states={prior.meta.n_states} pairs={prior.meta.n_pairs}")
    print(f"[ecd-prior] ridge={prior.meta.ridge:g} shrinkage={prior.meta.shrinkage:g}")
    return out


def main():
    """CLI entry point for fitting the ECD Gaussian Markov prior."""
    p = argparse.ArgumentParser()
    p.add_argument("--spec_csv", type=str, default="ogb_env_spec.csv")
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_states", type=int, default=500_000, help="Subsample states after loading; <=0 uses all.")
    p.add_argument("--max_pairs", type=int, default=500_000, help="Subsample adjacent pairs after loading; <=0 uses all.")
    p.add_argument("--ridge", type=float, default=1e-4, help="Clean-time covariance ridge before saving.")
    p.add_argument("--shrinkage", type=float, default=0.02, help="Shrink covariance toward its diagonal for stability.")
    args = p.parse_args()
    fit_prior(args)


if __name__ == "__main__":
    main()
