#!/usr/bin/env bash
# 8-bit attention at long context: GPT-124M (d=768, 6 heads of 128) on TinyStories with 4096-token
# sequences (several stories per sequence), 4 x 4096 tokens per step, 3000 steps, NVFP4 linears;
# BF16 SDPA vs nvfp4.attention, seeds 0 and 1.
set -uo pipefail
cd "$(dirname "$0")/.."
R=experiments/results/v0.9_long
mkdir -p $R
run() { name=$1; shift
  bash tools/run.sh train_$name experiments/train_gpt.py --steps 3000 --out $R/$name.jsonl "$@" > $R/$name.stdout 2>&1
  echo "$name exit=$? $(tail -2 $R/$name.stdout | head -1)"; sleep 60; }
common=(--precision nvfp4 --hp-last 2 --fuse-wgrad --fuse-producers --ctx 4096 --heads 6 --batch 4 --micro-batch 2)
for seed in 0 1; do
  run ctx4k_sdpa_s$seed  "${common[@]}" --attn sdpa --seed $seed
  run ctx4k_attn8_s$seed "${common[@]}" --attn 8bit --seed $seed
done
echo ALL_DONE
