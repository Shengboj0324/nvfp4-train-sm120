#!/usr/bin/env bash
# Run NVIDIA compute-sanitizer (memcheck, racecheck, synccheck, initcheck) over every kernel variant
# (tools/sanitize_drive.py: small ragged shapes), under the resource guard. Exit code != 0 on any finding.
#   tools/sanitize.sh [tool ...]
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ]; then
  if [ -x "$HOME/nvfp4/bin/python" ]; then PY=$HOME/nvfp4/bin/python; else PY=$(command -v python || command -v python3); fi
fi
if [ -z "${CS:-}" ]; then
  CS=$(command -v compute-sanitizer || ls /usr/local/cuda*/bin/compute-sanitizer /usr/local/cuda*/compute-sanitizer/compute-sanitizer 2>/dev/null | head -1)
fi
[ -x "$CS" ] || { echo "compute-sanitizer not found (set CS=...)"; exit 2; }
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
status=0
for tool in ${@:-memcheck racecheck synccheck initcheck}; do
  echo "=== $tool"
  # racecheck: judge this library's kernels only. PyTorch's own LayerNorm-backward kernel (reached via the
  # layers' backward) reports a shared-memory hazard warning that is not ours to fix.
  extra=()
  [ "$tool" = racecheck ] && extra=(--kernel-name-exclude kns=native)
  NVFP4_SANITIZE_TOOL=$tool "$PY" tools/guard.py --log "logs/sanitize_$tool.csv" -- \
    "$CS" --tool "$tool" "${extra[@]}" --print-limit 10 --error-exitcode 7 "$PY" tools/sanitize_drive.py 2>&1 \
    | grep -v "^\[guard\] sample" | grep -E "SUMMARY|DRIVE_OK|rror|hazard|Uninitialized|Invalid|Device Frame|guard\]"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && status=1
  echo "exit=$rc"
done
exit $status
