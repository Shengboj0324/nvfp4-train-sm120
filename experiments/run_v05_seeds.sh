#!/usr/bin/env bash
# v0.5 seed study: GPT-2-small (124M) BF16 vs NVFP4 over three seeds (init + data order).
# NVFP4 runs use the v0.5 code (GPU-drawn SR / RHT seeds, fused producers, fused wgrad accumulation);
# BF16 seed 0 is the v0.2 run (the BF16 code path is unchanged). Results: results/v0.5_seeds/.
set -uo pipefail
cd "$(dirname "$0")/.."
D=experiments/results/v0.5_seeds
mkdir -p "$D"
run() { name=$1; shift
  tools/run.sh "train_seeds_$name" experiments/train_gpt.py --out "$D/$name.jsonl" "$@" > "$D/$name.stdout" 2>&1
  echo "$name exit=$? $(tail -1 "$D/$name.stdout")"
  cp "logs/train_seeds_$name.csv" "$D/train_$name.csv"
  sleep 90; }  # cool down between runs
for f in bf16.jsonl bf16.stdout train_bf16.csv; do cp "experiments/results/v0.2/$f" "$D/${f/bf16/bf16_s0}"; done
for s in 0 1 2; do
  run "nvfp4_s$s" --precision nvfp4 --steps 3000 --hp-last 2 --fuse-wgrad --fuse-producers --seed "$s"
done
for s in 1 2; do
  run "bf16_s$s" --precision bf16 --steps 3000 --seed "$s"
done
echo ALL_DONE
