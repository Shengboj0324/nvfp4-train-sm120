#!/usr/bin/env bash
# NVFP4 MoE GPT (8 experts, top-2), 3000 steps, MoE layers on the expert-parallel path over a one-process group,
# return messages (expert outputs, input gradients) in bf16 vs MXFP8 (MoE.ep_wire_return): the loss effect.
# Seeds: $SEEDS (default "0 1"). Results: $RET_DIR (default experiments/results/v0.18_ep_return/).
set -uo pipefail
cd "$(dirname "$0")/.."
D=${RET_DIR:-experiments/results/v0.18_ep_return}
mkdir -p "$D"
run() { name=$1; seed=$2; shift 2
  tools/run.sh "ret_$name" experiments/train_gpt.py --out "$D/$name.jsonl" --steps "${STEPS:-3000}" --moe-experts 8 \
    --seed "$seed" --precision nvfp4 --hp-last 2 --fuse-wgrad --fuse-producers --moe-ep-wire bf16 "$@" > "$D/$name.stdout" 2>&1
  echo "$name exit=$? $(grep val_loss "$D/$name.stdout" | tail -1)"; sleep 60; }
for s in ${SEEDS:-0 1}; do
  for r in bf16 fp8; do
    run "ret_${r}_s$s" "$s" --moe-ep-wire-return "$r"
  done
done
echo RET_DONE
