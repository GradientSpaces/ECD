#!/bin/bash
# Shared setup for all ECD scripts.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLCONFIGDIR=/tmp/matplotlib

ENV="${1:?usage: bash scripts/<name>.sh <env-name> [seed]}"
SEED="${2:-0}"

# humanoid planners were trained with a larger batch in the paper
BATCH=128; case "$ENV" in humanoidmaze*) BATCH=192;; esac
