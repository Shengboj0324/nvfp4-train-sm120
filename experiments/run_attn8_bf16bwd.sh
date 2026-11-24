#!/usr/bin/env bash
# 8-bit attention forward + exact flash backward of it (attention(backward="bf16")) at 4096 tokens, same setup
# and seeds as run_attn8_long.sh / run_attn8_long_v2.sh (results in experiments/results/v0.9_long).
set -uo pipefail
cd "$(dirname "$0")/.."
R=experiments/results/v0.9_long
run() { name=$1; shift
  bash tools/run.sh train_$name experiments/train_gpt.py --steps 3000 --out $R/$name.jsonl "$@" > $R/$name.stdout 2>&1
  echo "$name exit=$? $(tail -2 $R/$name.stdout | head -1)"; sleep 60; }
common=(--precision nvfp4 --hp-last 2 --fuse-wgrad --fuse-producers --ctx 4096 --heads 6 --batch 4 --micro-batch 2)
for seed in 0 1; do
  run ctx4k_attn8bf16bwd_s$seed "${common[@]}" --attn 8bit --attn-backward bf16 --seed $seed
done
echo ALL_DONE
