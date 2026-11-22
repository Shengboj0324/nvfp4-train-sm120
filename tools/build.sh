#!/usr/bin/env bash
# Build the CUDA extension in place (WSL2/Linux). Usage: tools/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$CUDA_HOME/bin:$PATH
export MAX_JOBS=${MAX_JOBS:-8}
PY=${PY:-$HOME/nvfp4/bin/python}
exec "$PY" tools/guard.py --log logs/build.csv --max-load-per-cpu 3 -- "$PY" setup.py build_ext --inplace
