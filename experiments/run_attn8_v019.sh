#!/usr/bin/env bash
# v0.19: GPT-124M at 4096 tokens (6 heads of 128), NVFP4 linears, 3000 steps: BF16 SDPA vs the 8-bit attention with
# the exact flash backward (backward="bf16") vs the v0.19 8-bit backward (FP16 dQ / dK / dV, consistent Delta).
# Seeds: $SEEDS (default "0 1"). Results: experiments/results/v0.19_attn/.
set -uo pipefail
cd "$(dirname "$0")/.."
R=experiments/results/v0.19_attn
mkdir -p $R
run() { name=$1; shift
  bash tools/run.sh train_$name experiments/train_gpt.py --steps 3000 --out $R/$name.jsonl "$@" > $R/$name.stdout 2>&1
  echo "$name exit=$? $(grep val_loss $R/$name.stdout | tail -1)"; sleep 60; }
common=(--precision nvfp4 --hp-last 2 --fuse-wgrad --fuse-producers --ctx 4096 --heads 6 --batch 4 --micro-batch 2)
for seed in ${SEEDS:-0 1}; do
  run ctx4k_attn8bwd8_s$seed "${common[@]}" --attn 8bit --attn-backward 8bit --seed $seed
  run ctx4k_sdpa_s$seed "${common[@]}" --seed $seed
  run ctx4k_attn8bwdbf16_s$seed "${common[@]}" --attn 8bit --attn-backward bf16 --seed $seed
done
echo ALL_DONE
