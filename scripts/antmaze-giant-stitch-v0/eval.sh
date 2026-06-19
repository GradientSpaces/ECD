#!/bin/bash
# Evaluate CD / ECD / CDGS on antmaze-giant-stitch-v0.  Usage:  bash eval.sh <cd|ecd|cdgs> <seed>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLCONFIGDIR=/tmp/matplotlib
METHOD="${1:?usage: bash eval.sh <cd|ecd|cdgs> <seed>}"; SEED="${2:-0}"; ENV=antmaze-giant-stitch-v0

COMMON=(--env "$ENV" --planner_name planner --planner_epoch latest --invdyn_name invdyn --invdyn_epoch latest --seed "$SEED" \
        --ev_n_comp 9 --n_max_steps 2000 --repl_max_n 10 \
        --b_size_per_prob 40 --is_replan ada_dist)

case "$METHOD" in
  cd)    # CompDiffuser baseline: interleave stitching, map-free overlap ranking
    python -m ecd.eval "${COMMON[@]}" --eval_name cd --ev_cp_infer_t_type interleave ;;
  ecd)   # ECD (Ours): laplacian reaction, map-free overlap ranking
    python -m ecd.eval "${COMMON[@]}" --eval_name ecd --ev_cp_infer_t_type ecd_chunk \
      --ecd_base_scale 0.5 --ecd_react_scale 0.1 --ecd_react_clip 1.0 --ecd_chunk_react_type markov --ecd_markov_type laplacian --ecd_markov_rho 0.25 --ecd_rank_type overlap ;;
  cdgs)  # CDGS baseline: compute-scaled GSC with U=4 resampling rounds (~4x the planning compute)
    python -m ecd.eval "${COMMON[@]}" --eval_name cdgs --ev_cp_infer_t_type gsc_resampling --gsc_u 4 ;;
  *) echo "unknown method '$METHOD' (use: cd | ecd | cdgs)" >&2; exit 1 ;;
esac
