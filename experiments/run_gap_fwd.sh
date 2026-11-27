#!/usr/bin/env bash
# v0.19: forward formats that might close the NVFP4 loss gap (experiments/gap_fwd.py, emulated forward, the recipe's
# backward). Configurations: $CFGS (names below), seeds: $SEEDS (default "0 1"). Results: experiments/results/v0.19_gap/.
set -uo pipefail
cd "$(dirname "$0")/.."
R=experiments/results/v0.19_gap
mkdir -p $R
run() { name=$1; shift
  bash tools/run.sh gapf_$name experiments/gap_fwd.py --precision nvfp4 --hp-last 2 --steps 3000 \
    --out $R/$name.jsonl "$@" > $R/$name.stdout 2>&1
  echo "$name exit=$? $(grep val_loss $R/$name.stdout | tail -1) $(grep eval_bf16 $R/$name.stdout | tail -1)"; sleep 60; }
args() { case $1 in
  ctrl) echo "" ;;                       # the recipe, emulated
  rot16) echo "--fwd-rot 16" ;;
  rot128) echo "--fwd-rot 128" ;;
  xmx8) echo "--fwd-x mxfp8" ;;          # W4A8: MXFP8 activations, NVFP4 weights
  w1d) echo "--fwd-w nvfp4_1d" ;;
  x2) echo "--fwd-x nvfp4x2" ;;          # two NVFP4 terms for the activations
  *) echo "unknown $1" >&2; exit 2 ;;
esac; }
for s in ${SEEDS:-0 1}; do
  for cfg in ${CFGS:-ctrl rot128 xmx8}; do
    run fwd_${cfg}_s$s --seed $s $(args $cfg)
  done
done
echo FWD_DONE
