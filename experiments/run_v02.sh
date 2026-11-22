#!/usr/bin/env bash
# v0.2 end-to-end runs (fused dual quantizer, saved NVFP4 activations):
#   results/v0.2/        GPT-2-small (124M, d=768): BF16 vs NVFP4, same config as the v0.1 main run
#   results/v0.2_wide/   wide GPT (~500M, d=2048, 8 layers): BF16 vs NVFP4
set -uo pipefail
cd "$(dirname "$0")/.."
run() { dir=$1 name=$2; shift 2
  mkdir -p "$dir"
  tools/run.sh "train_$(basename "$dir")_$name" experiments/train_gpt.py --out "$dir/$name.jsonl" "$@" > "$dir/$name.stdout" 2>&1
  echo "$dir/$name exit=$? $(tail -1 "$dir/$name.stdout")"
  cp "logs/train_$(basename "$dir")_$name.csv" "$dir/train_$name.csv"
  sleep 90; }  # cool down between runs
S=experiments/results/v0.2
W=experiments/results/v0.2_wide
WIDE="--d 2048 --layers 8 --heads 16 --steps 1500 --lr 3e-4 --warmup 150 --micro-batch 8"
run $S bf16  --precision bf16  --steps 3000
run $S nvfp4 --precision nvfp4 --steps 3000 --hp-last 2 --fuse-wgrad
run $W bf16  --precision bf16  $WIDE
run $W nvfp4 --precision nvfp4 $WIDE --hp-last 1 --fuse-wgrad
echo ALL_DONE
