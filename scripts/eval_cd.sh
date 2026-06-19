#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# CompDiffuser (CD) baseline: interleave inference, overlap-distance candidate selection (no maze map).

INV=(--invdyn_name invdyn --invdyn_epoch latest); case "$ENV" in pointmaze*) INV=();; esac

python -m ecd.eval --env "$ENV" --planner_name planner --planner_epoch latest "${INV[@]}" \
  --seed "$SEED" --ev_cp_infer_t_type interleave \
  --b_size_per_prob 40 --is_replan ada_dist --eval_name cd
