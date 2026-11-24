#!/usr/bin/env bash
# Three-seed ablation of an NVFP4 recipe change on GPT-124M (TinyStories, 3000 steps, 512 tokens):
#   experiments/run_ablation.sh NAME [extra train_gpt.py flags...]
# Runs NAME_s{0,1,2} into experiments/results/v0.10_ablate/ with the standard NVFP4 setup plus the flags;
# experiments/ablation_summary.py compares each run with the BF16 run of the same seed (v0.5_seeds).
set -uo pipefail
cd "$(dirname "$0")/.."
D=experiments/results/v0.10_ablate
mkdir -p "$D"
name=$1; shift
for s in ${SEEDS:-0 1 2}; do
  tools/run.sh "ablate_${name}_s$s" experiments/train_gpt.py --out "$D/${name}_s$s.jsonl" --precision nvfp4 \
    --steps 3000 --hp-last 2 --fuse-wgrad --fuse-producers --seed "$s" "$@" > "$D/${name}_s$s.stdout" 2>&1
  echo "${name}_s$s exit=$? $(grep val_loss "$D/${name}_s$s.stdout" | tail -1)"
  sleep 60
done
echo "ABLATION_DONE $name"
