#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Fit the data-only Gaussian-Markov transition prior used by the ECD reaction term.
mkdir -p "logs/$ENV/ecd_prior"
python -m ecd.fit_ecd_prior --env "$ENV" \
  --out "logs/$ENV/ecd_prior/gaussian_markov.pt" \
  --max_states 500000 --max_pairs 500000 --ridge 1e-4 --shrinkage 0.02
