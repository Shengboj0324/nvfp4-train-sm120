#!/usr/bin/env bash
# 8-bit attention convergence check: GPT-124M on TinyStories, 3000 steps, seed 0 (compare with
# experiments/results/v0.8/nvfp4_s0 and the BF16 baseline of docs/RESULTS.md).
set -uo pipefail
cd "$(dirname "$0")/.."
R=experiments/results/v0.9
mkdir -p $R
run() { name=$1; shift
  bash tools/run.sh train_$name experiments/train_gpt.py --steps 3000 --out $R/$name.jsonl "$@" > $R/$name.stdout 2>&1
  echo "$name exit=$? $(tail -2 $R/$name.stdout | head -1)"; sleep 60; }
run nvfp4_attn8_s0 --precision nvfp4 --hp-last 2 --fuse-wgrad --fuse-producers --attn 8bit
run bf16_attn8_s0  --precision bf16 --attn 8bit
echo ALL_DONE
