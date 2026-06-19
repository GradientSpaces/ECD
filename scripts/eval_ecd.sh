#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ECD
INV=(--invdyn_name invdyn --invdyn_epoch latest); case "$ENV" in pointmaze*) INV=();; esac

python -m ecd.eval --env "$ENV" --planner_name planner --planner_epoch latest "${INV[@]}" \
  --seed "$SEED" --ev_cp_infer_t_type ecd_chunk \
  --b_size_per_prob 40 --is_replan ada_dist --eval_name ecd \
  --ecd_base_scale 0.5 --ecd_react_scale 0.1 --ecd_react_clip 1.0 \
  --ecd_chunk_react_type markov --ecd_markov_type laplacian --ecd_markov_rho 0.25 \
  --ecd_rank_type overlap
