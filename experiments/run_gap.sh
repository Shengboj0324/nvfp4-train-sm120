#!/usr/bin/env bash
# v0.19: where the NVFP4 loss gap of GPT-124M comes from (experiments/gap_ablate.py): the unfused NVFP4Linear path
# with chosen GEMMs in BF16, or the output gradient rounded to nearest. Seeds: $SEEDS (default "0"); configurations:
# $CFGS, each base, rtn, a --gap-bf16 entry, or ENTRY.LAYER (that entry for one of qkv / proj / fc1 / fc2 only).
# Results: experiments/results/v0.19_gap/.
set -uo pipefail
cd "$(dirname "$0")/.."
R=experiments/results/v0.19_gap
mkdir -p $R
run() { name=$1; shift
  bash tools/run.sh gap_$name experiments/gap_ablate.py --precision nvfp4 --hp-last 2 --steps 3000 \
    --out $R/$name.jsonl "$@" > $R/$name.stdout 2>&1
  echo "$name exit=$? $(grep val_loss $R/$name.stdout | tail -1)"; sleep 60; }
for s in ${SEEDS:-0}; do
  for cfg in ${CFGS:-base fprop dgrad wgrad rtn}; do
    case $cfg in
      base) run base_s$s --seed $s ;;
      rtn) run rtn_s$s --seed $s --gap-rtn ;;
      *.*) run ${cfg/./_}_s$s --seed $s --gap-bf16 ${cfg%.*} --gap-only ${cfg#*.} ;;
      *) run ${cfg}_s$s --seed $s --gap-bf16 $cfg ;;
    esac
  done
done
echo GAP_DONE
