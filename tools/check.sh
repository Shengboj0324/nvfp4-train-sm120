#!/usr/bin/env bash
# Local verification on the GPU machine: build, run the full test suite, optionally run benchmarks.
#   tools/check.sh            build + tests
#   tools/check.sh bench      build + tests + GEMM/quantizer benchmarks
set -uo pipefail
cd "$(dirname "$0")/.."
tools/build.sh 2>&1 | grep -E "error|\[guard\]" || true
[ -f nvfp4/_C*.so ] || ls nvfp4/_C*.so >/dev/null || { echo "build failed"; exit 1; }
tools/run.sh tests -m pytest -q tests/ -p no:cacheprovider 2>&1 | tail -3
if [ "${1:-}" = "bench" ]; then
  tools/run.sh bench_quantize bench/bench_quantize.py 2>&1 | tail -17
  tools/run.sh bench_gemm bench/bench_gemm.py --iters 20 2>&1 | tail -11
fi
