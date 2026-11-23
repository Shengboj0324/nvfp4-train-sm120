#!/usr/bin/env bash
# One-shot validation session on a rented multi-GPU Linux machine (e.g. a Runpod pod with 2x RTX 5090),
# covering what a single laptop GPU cannot: NCCL across real GPUs, and a second sm_120 GPU model.
#
#   curl -sSL https://raw.githubusercontent.com/Shengboj0324/nvfp4-train-sm120/main/tools/pod_session.sh | bash
#   (or: git clone ... && bash tools/pod_session.sh)
#
# Phases (each logged and timed; a failing phase is recorded and the session continues):
#   env     GPUs, driver, topology, CUDA toolkit (installs CUDA 13.0 / 12.8 if no nvcc >= 12.8), PyTorch
#   build   the extension for sm_120a
#   tests   the full suite (incl. the two-GPU NCCL tests), then sanitizers + 5-minute soak
#   bench   GEMM, quantizer, GPT-block and Llama-block benchmarks on this GPU model
#   gpt     GPT-124M, 3000 steps: BF16 and NVFP4 on one GPU, NVFP4 with DDP over all GPUs (NCCL)
# Everything ends up in $OUT (default /workspace/nvfp4_results) and $OUT.tar.gz.
# Every GPU job runs under tools/guard.py (temperature / memory limits, over all GPUs).
set -uo pipefail
OUT=${OUT:-/workspace/nvfp4_results}
REPO=${REPO:-/workspace/nvfp4-train-sm120}
PHASES=${PHASES:-env build tests bench gpt}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/session.log") 2>&1
echo "== session start $(date -u +%FT%TZ)"
declare -A STATUS

phase() {  # phase <name> <function>
  local t0=$SECONDS
  echo; echo "==================== $1 ===================="
  if "$2"; then STATUS[$1]=ok; else STATUS[$1]=FAILED; fi
  echo "== $1: ${STATUS[$1]} ($((SECONDS - t0)) s)"
}

ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }  # ver_ge A B: A >= B

do_env() {
  nvidia-smi || return 1
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total,power.limit,clocks.max.sm,pcie.link.gen.max,pcie.link.width.max --format=csv | tee "$OUT/gpus.csv"
  nvidia-smi topo -m | tee "$OUT/topology.txt"
  DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+')
  echo "driver supports CUDA $DRIVER_CUDA"
  if ! command -v nvcc >/dev/null; then
    for d in /usr/local/cuda-13.0 /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.8; do
      [ -x "$d/bin/nvcc" ] && export PATH="$d/bin:$PATH" && break
    done
  fi
  NVCC=$(command -v nvcc && nvcc --version | grep -oP 'release \K[0-9.]+' || echo 0)
  NVCC=${NVCC##*$'\n'}
  if ! ver_ge "$NVCC" 12.8; then
    echo "no nvcc >= 12.8 (found $NVCC); installing a CUDA toolkit"
    . /etc/os-release
    repo="ubuntu${VERSION_ID//./}"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/keyring.deb
    dpkg -i /tmp/keyring.deb && apt-get update -qq
    if ver_ge "$DRIVER_CUDA" 13.0; then pkg=cuda-toolkit-13-0; d=/usr/local/cuda-13.0; else pkg=cuda-toolkit-12-8; d=/usr/local/cuda-12.8; fi
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$pkg" || return 1
    export PATH="$d/bin:$PATH"
    NVCC=$(nvcc --version | grep -oP 'release \K[0-9.]+')
  fi
  export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
  echo "nvcc $NVCC at $CUDA_HOME"
  command -v ninja >/dev/null || pip install -q ninja
  if ver_ge "$NVCC" 13.0; then idx=cu130; else idx=cu128; fi
  # keep a preinstalled PyTorch only if it is >= 2.7, has sm_120 kernels and the same CUDA major as nvcc
  # (torch.utils.cpp_extension refuses to build across CUDA major versions)
  NVCC=$NVCC python - <<'PY' || pip install -q "torch==2.14.1" --index-url "https://download.pytorch.org/whl/$idx" \
                   || pip install -q "torch>=2.7" --index-url "https://download.pytorch.org/whl/$idx" || return 1
import os, sys
try:
    import torch
except ImportError:
    sys.exit(1)
ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
same_major = (torch.version.cuda or "0").split(".")[0] == os.environ["NVCC"].split(".")[0]
ok = ver >= (2, 7) and same_major and "sm_120" in torch.cuda.get_arch_list()
print("preinstalled torch", torch.__version__, torch.version.cuda, "ok" if ok else "-> replacing")
sys.exit(0 if ok else 1)
PY
  pip install -q numpy pytest matplotlib tiktoken
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" | tee "$OUT/torch.txt"
  if [ ! -d "$REPO/.git" ]; then git clone -q https://github.com/Shengboj0324/nvfp4-train-sm120 "$REPO" || return 1; fi
  git -C "$REPO" log --oneline -1 | tee "$OUT/commit.txt"
  echo "export PATH=$PATH CUDA_HOME=$CUDA_HOME" > "$OUT/env.sh"
}

do_build() {
  cd "$REPO" && MAX_JOBS=${MAX_JOBS:-8} python tools/guard.py --log "$OUT/build.csv" -- \
    python setup.py build_ext --inplace > "$OUT/build.log" 2>&1
  local rc=$?
  grep -E "error|registers|spill" "$OUT/build.log" | grep -v "0 bytes spill" | tail -20
  ls nvfp4/_C*.so >/dev/null 2>&1 && [ $rc -eq 0 ]
}

do_tests() {
  cd "$REPO" || return 1
  export PY=python PYTHONPATH=$REPO TORCHINDUCTOR_COMPILE_THREADS=4
  local st=0
  timeout 1800 python tools/guard.py --log "$OUT/tests.csv" -- python -m pytest -q tests/ -p no:cacheprovider -rs \
    > "$OUT/tests.log" 2>&1 || st=1
  tail -15 "$OUT/tests.log"
  # NCCL between GeForce cards: if the two-GPU tests failed, retry once without peer-to-peer (some
  # consumer-GPU hosts report P2P that hangs) and record which configuration worked.
  if grep -q "FAILED.*multi_gpu_nccl" "$OUT/tests.log" || ! grep -qE "[0-9]+ passed" "$OUT/tests.log"; then
    NCCL_P2P_DISABLE=1 timeout 900 python -m pytest -q tests/test_integration_cuda.py -k multi_gpu_nccl -p no:cacheprovider -rs \
      > "$OUT/tests_nccl_p2p_off.log" 2>&1
    echo "retry with NCCL_P2P_DISABLE=1: $(tail -1 "$OUT/tests_nccl_p2p_off.log")"
  fi
  timeout 300 python -m pytest -q tests/test_integration_cuda.py -k multi_gpu_nccl -p no:cacheprovider -rs -v \
    > "$OUT/tests_nccl_verbose.log" 2>&1
  grep -E "PASSED|FAILED|SKIPPED|passed|failed" "$OUT/tests_nccl_verbose.log" | tail -8
  bash tools/sanitize.sh > "$OUT/sanitize.log" 2>&1 || st=1
  grep -E "===|SUMMARY|exit=" "$OUT/sanitize.log"
  timeout 900 python tools/guard.py --log "$OUT/soak.csv" -- python tools/soak.py --minutes "${SOAK_MINUTES:-5}" \
    > "$OUT/soak.log" 2>&1 || st=1
  grep -E "SOAK_OK|FAIL" "$OUT/soak.log"
  return $st
}

do_bench() {
  cd "$REPO" || return 1
  export PYTHONPATH=$REPO
  local st=0
  python tools/guard.py --log "$OUT/bench_gemm.csv" -- python bench/bench_gemm.py --cooldown 0.5 --out "$OUT/gemm.json" \
    > "$OUT/bench_gemm.log" 2>&1 || st=1
  tail -16 "$OUT/bench_gemm.log"
  python tools/guard.py --log "$OUT/bench_quantize.csv" -- python bench/bench_quantize.py --out "$OUT/quantize.json" \
    > "$OUT/bench_quantize.log" 2>&1 || st=1
  tail -10 "$OUT/bench_quantize.log"
  python tools/guard.py --log "$OUT/bench_train_step.csv" -- python bench/bench_train_step.py --out "$OUT/train_step.json" \
    > "$OUT/bench_train_step.log" 2>&1 || st=1
  tail -8 "$OUT/bench_train_step.log"
  python tools/guard.py --log "$OUT/bench_llama.csv" -- python bench/bench_llama_block.py --out "$OUT/llama_block.json" \
    > "$OUT/bench_llama.log" 2>&1 || st=1
  tail -6 "$OUT/bench_llama.log"
  return $st
}

do_gpt() {
  cd "$REPO" || return 1
  export PYTHONPATH=$REPO NVFP4_DATA=${NVFP4_DATA:-/workspace/nvfp4_data}
  [ -f "$NVFP4_DATA/train.bin" ] || python experiments/prepare_tinystories.py --train-mb 250 --out "$NVFP4_DATA" || return 1
  local st=0 ngpu
  ngpu=$(nvidia-smi -L | wc -l)
  local steps=${GPT_STEPS:-3000}
  local common=(--precision nvfp4 --steps "$steps" --hp-last 2 --fuse-wgrad --fuse-producers --seed 0)
  python tools/guard.py --log "$OUT/gpt_nvfp4.csv" -- python experiments/train_gpt.py "${common[@]}" \
    --out "$OUT/gpt_nvfp4_1gpu.jsonl" > "$OUT/gpt_nvfp4_1gpu.log" 2>&1 || st=1
  python tools/guard.py --log "$OUT/gpt_bf16.csv" -- python experiments/train_gpt.py --precision bf16 --steps "$steps" --seed 0 \
    --out "$OUT/gpt_bf16_1gpu.jsonl" > "$OUT/gpt_bf16_1gpu.log" 2>&1 || st=1
  if [ "$ngpu" -ge 2 ]; then
    # (torchrun would take a bare "--d" for one of its own options; the defaults are the 124M model)
    python tools/guard.py --log "$OUT/gpt_ddp.csv" -- torchrun --standalone --nproc_per_node "$ngpu" \
      experiments/train_gpt.py "${common[@]}" --out "$OUT/gpt_nvfp4_ddp${ngpu}.jsonl" \
      > "$OUT/gpt_nvfp4_ddp.log" 2>&1 || st=1
  fi
  for f in "$OUT"/gpt_*.jsonl; do
    python - "$f" <<'PY'
import json, statistics, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
val = [r for r in recs if "val_loss" in r]
tps = [r["tok_per_s"] for r in recs if r.get("tok_per_s")]
print(f"{sys.argv[1].split('/')[-1]:28s} final val {val[-1]['val_loss']:.4f} at step {val[-1]['step']}, "
      f"median tok/s {statistics.median(tps[len(tps)//2:]):,.0f}")
PY
  done
  return $st
}

for p in $PHASES; do
  case $p in
    env) phase env do_env ;;
    build) phase build do_build ;;
    tests) phase tests do_tests ;;
    bench) phase bench do_bench ;;
    gpt) phase gpt do_gpt ;;
  esac
done
echo; echo "== summary"; for p in $PHASES; do echo "  $p: ${STATUS[$p]:-not run}"; done
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")" && echo "results: $OUT.tar.gz"
echo "== session end $(date -u +%FT%TZ)"
