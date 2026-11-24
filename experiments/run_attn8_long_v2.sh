#!/usr/bin/env bash
# Rerun the 8-bit-attention side of run_attn8_long.sh after the dQ fix (BF16 dS K, consistent Delta).
set -uo pipefail
cd "$(dirname "$0")/.."
R=experiments/results/v0.9_long
run() { name=$1; shift
  bash tools/run.sh train_$name experiments/train_gpt.py --steps 3000 --out $R/$name.jsonl "$@" > $R/$name.stdout 2>&1
  echo "$name exit=$? $(tail -2 $R/$name.stdout | head -1)"; sleep 60; }
common=(--precision nvfp4 --hp-last 2 --fuse-wgrad --fuse-producers --ctx 4096 --heads 6 --batch 4 --micro-batch 2)
for seed in 0 1; do
  run ctx4k_attn8fix_s$seed "${common[@]}" --attn 8bit --seed $seed
done
echo ALL_DONE
