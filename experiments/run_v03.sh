#!/usr/bin/env bash
# v0.3 end-to-end runs (TMA GEMM, split-K, LayerNorm / GELU fused into quantization). The BF16
# baselines are the v0.2 runs, copied in: the BF16 code path did not change.
#   results/v0.3/        GPT-2-small (124M, d=768)
#   results/v0.3_wide/   wide GPT (507M, d=2048, 8 layers)
set -uo pipefail
cd "$(dirname "$0")/.."
run() { dir=$1 name=$2; shift 2
  mkdir -p "$dir"
  tools/run.sh "train_$(basename "$dir")_$name" experiments/train_gpt.py --out "$dir/$name.jsonl" "$@" > "$dir/$name.stdout" 2>&1
  echo "$dir/$name exit=$? $(tail -1 "$dir/$name.stdout")"
  cp "logs/train_$(basename "$dir")_$name.csv" "$dir/train_$name.csv"
  sleep 90; }  # cool down between runs
S=experiments/results/v0.3
W=experiments/results/v0.3_wide
WIDE="--d 2048 --layers 8 --heads 16 --steps 1500 --lr 3e-4 --warmup 150 --micro-batch 8"
mkdir -p $S $W
for f in bf16.jsonl bf16.stdout train_bf16.csv; do
  cp experiments/results/v0.2/$f $S/$f
  cp experiments/results/v0.2_wide/$f $W/$f
done
run $S nvfp4 --precision nvfp4 --steps 3000 --hp-last 2 --fuse-wgrad --fuse-producers
run $W nvfp4 --precision nvfp4 $WIDE --hp-last 1 --fuse-wgrad --fuse-producers
echo ALL_DONE
