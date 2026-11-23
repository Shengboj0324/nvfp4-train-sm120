#!/usr/bin/env bash
# Local verification on the GPU machine (everything runs under tools/guard.py).
#   tools/check.sh            build + tests
#   tools/check.sh bench      build + tests + GEMM / quantizer benchmarks
#   tools/check.sh release    build + tests + compute-sanitizer (4 tools) + 5-minute soak test
set -uo pipefail
cd "$(dirname "$0")/.."
tools/build.sh 2>&1 | grep -E "error|\[guard\]" || true
ls nvfp4/_C*.so >/dev/null 2>&1 || { echo "build failed"; exit 1; }
status=0
tools/run.sh tests -m pytest -q tests/ -p no:cacheprovider 2>&1 | grep -v "^\[guard\] sample" | tail -3
[ "${PIPESTATUS[0]}" -eq 0 ] || status=1
case "${1:-}" in
  bench)
    tools/run.sh bench_quantize bench/bench_quantize.py 2>&1 | tail -17
    tools/run.sh bench_gemm bench/bench_gemm.py --iters 20 2>&1 | tail -13 ;;
  release)
    tools/sanitize.sh || status=1
    tools/run.sh soak tools/soak.py --minutes "${SOAK_MINUTES:-5}" 2>&1 | grep -v "^\[guard\] sample" | tail -3
    [ "${PIPESTATUS[0]}" -eq 0 ] || status=1 ;;
esac
[ $status -eq 0 ] && echo "CHECK PASSED" || echo "CHECK FAILED"
exit $status
