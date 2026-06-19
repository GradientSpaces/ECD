#!/bin/bash
# train (planner) pointmaze-giant-stitch-v0  (auto-generated per-env recipe)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLCONFIGDIR=/tmp/matplotlib
SEED="${1:-0}"; ENV=pointmaze-giant-stitch-v0

# 1) short-horizon diffusion planner
python -m ecd.train --env "$ENV" --planner_name planner --seed "$SEED" --batch_size 128
