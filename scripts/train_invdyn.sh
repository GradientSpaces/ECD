#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Train the inverse-dynamics model.
# Not required for pointmaze.
python -m ecd.invdyn --env "$ENV" --invdyn_name invdyn --seed "$SEED"
