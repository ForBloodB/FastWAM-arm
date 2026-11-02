#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/aup/lerobot-fastwam
export LD_LIBRARY_PATH="$ROOT/rocm72-root/opt/rocm-7.2.4/lib:$ROOT/venv-rocm72-fwam/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="$ROOT/vendor/FastWAM-main/src:$ROOT/host:$ROOT/kr260:$ROOT/common${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/venv-rocm72-fwam/bin/python" "$@"
