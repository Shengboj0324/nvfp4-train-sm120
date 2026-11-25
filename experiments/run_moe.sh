#!/usr/bin/env bash
# MoE GPT (8 experts, top-2, expert hidden 4d/2 = 1536: compute-matched to the dense MLP) on TinyStories, 3000
# steps, BF16 vs NVFP4 experts and linears (last 2 blocks BF16). Seeds: $SEEDS (default "0 1"; seed 0 runs are
# named moe8_{nvfp4,bf16}, others get an _s<seed> suffix). Results: $MOE_DIR (default experiments/results/v0.12_moe/).
set -uo pipefail
cd "$(dirname "$0")/.."
D=${MOE_DIR:-experiments/results/v0.12_moe}
mkdir -p "$D"
run() { name=$1; seed=$2; shift 2
  tools/run.sh "moe_$name" experiments/train_gpt.py --out "$D/$name.jsonl" --steps 3000 --moe-experts 8 --seed "$seed" "$@" \
    > "$D/$name.stdout" 2>&1
  echo "$name exit=$? $(grep val_loss "$D/$name.stdout" | tail -1)"; sleep 60; }
for s in ${SEEDS:-0 1}; do
  sfx=$([ "$s" = 0 ] || echo "_s$s")
  run "moe8_nvfp4$sfx" "$s" --precision nvfp4 --hp-last 2 --fuse-wgrad --fuse-producers
  run "moe8_bf16$sfx" "$s" --precision bf16
done
echo MOE_DONE
