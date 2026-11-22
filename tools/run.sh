#!/usr/bin/env bash
# Run a Python command from the repo root under the resource guard. Usage: tools/run.sh <logname> <args...>
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/nvfp4/bin/python}
name=$1; shift
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
exec "$PY" tools/guard.py --log "logs/$name.csv" -- "$PY" "$@"
