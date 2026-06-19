#!/bin/bash
# train (planner + invdyn + ecd_prior) antmaze-large-stitch-v0-o15d  (auto-generated per-env recipe)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLCONFIGDIR=/tmp/matplotlib
SEED="${1:-0}"; ENV=antmaze-large-stitch-v0-o15d

# 1) short-horizon diffusion planner
python -m ecd.train --env "$ENV" --planner_name planner --seed "$SEED" --batch_size 128

# 2) inverse-dynamics model (x-y waypoints -> robot actions)
python -m ecd.invdyn --env "$ENV" --invdyn_name invdyn --seed "$SEED"

# 3) data-only Gaussian-Markov transition prior (for the fitted reaction term)
mkdir -p "logs/$ENV/ecd_prior"
python -m ecd.fit_ecd_prior --env "$ENV" --out "logs/$ENV/ecd_prior/gaussian_markov.pt" \
  --max_states 500000 --max_pairs 500000 --ridge 1e-4 --shrinkage 0.02
