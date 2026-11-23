#!/usr/bin/env bash
# Run NVIDIA compute-sanitizer (memcheck, racecheck, synccheck, initcheck) over every kernel variant
# (tools/sanitize_drive.py: small ragged shapes), under the resource guard. Exit code != 0 on any finding.
#   tools/sanitize.sh [tool ...]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/nvfp4/bin/python}
CS=${CS:-$(command -v compute-sanitizer || echo /usr/local/cuda-13.0/bin/compute-sanitizer)}
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
status=0
for tool in ${@:-memcheck racecheck synccheck initcheck}; do
  echo "=== $tool"
  "$PY" tools/guard.py --log "logs/sanitize_$tool.csv" -- \
    "$CS" --tool "$tool" --print-limit 10 --error-exitcode 7 "$PY" tools/sanitize_drive.py 2>&1 \
    | grep -v "^\[guard\] sample" | grep -E "SUMMARY|DRIVE_OK|rror|hazard|Uninitialized|Invalid|Device Frame|guard\]"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && status=1
  echo "exit=$rc"
done
exit $status
