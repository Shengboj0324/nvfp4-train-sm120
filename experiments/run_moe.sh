#!/usr/bin/env bash
# MoE GPT (8 experts, top-2, expert hidden 4d/2 = 1536: compute-matched to the dense MLP) on TinyStories, 3000
# steps, BF16 vs NVFP4 experts and linears (last 2 blocks BF16), seed 0. Results: experiments/results/v0.10_moe/.
set -uo pipefail
cd "$(dirname "$0")/.."
D=experiments/results/v0.10_moe
mkdir -p "$D"
run() { name=$1; shift
  tools/run.sh "moe_$name" experiments/train_gpt.py --out "$D/$name.jsonl" --steps 3000 --moe-experts 8 --seed 0 "$@" \
    > "$D/$name.stdout" 2>&1
  echo "$name exit=$? $(grep val_loss "$D/$name.stdout" | tail -1)"; sleep 60; }
run moe8_bf16 --precision bf16
run moe8_nvfp4 --precision nvfp4 --hp-last 2 --fuse-wgrad --fuse-producers
echo MOE_DONE
