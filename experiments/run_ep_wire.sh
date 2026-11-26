#!/usr/bin/env bash
# NVFP4 MoE GPT (8 experts, top-2) on TinyStories, 3000 steps, its MoE layers on the expert-parallel path over a
# one-process group, with bf16 / NVFP4 activation / NVFP4 activation + gradient messages (MoE.ep_wire,
# ep_wire_grad): the wire formats' effect on the loss. Seeds: $SEEDS (default "0 1"). Results: $WIRE_DIR
# (default experiments/results/v0.17_ep_wire/).
set -uo pipefail
cd "$(dirname "$0")/.."
D=${WIRE_DIR:-experiments/results/v0.17_ep_wire}
mkdir -p "$D"
run() { name=$1; seed=$2; shift 2
  tools/run.sh "wire_$name" experiments/train_gpt.py --out "$D/$name.jsonl" --steps "${STEPS:-3000}" --moe-experts 8 \
    --seed "$seed" --precision nvfp4 --hp-last 2 --fuse-wgrad --fuse-producers "$@" > "$D/$name.stdout" 2>&1
  echo "$name exit=$? $(grep val_loss "$D/$name.stdout" | tail -1)"; sleep 60; }
for s in ${SEEDS:-0 1}; do
  for w in bf16 nvfp4 nvfp4+grad; do
    run "wire_${w/+/_}_s$s" "$s" --moe-ep-wire "$w"
  done
done
echo WIRE_DONE
