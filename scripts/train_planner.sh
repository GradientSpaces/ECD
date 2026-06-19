#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Train the short-horizon diffusion planner.
python -m ecd.train --env "$ENV" --planner_name planner --seed "$SEED" --batch_size "$BATCH"
