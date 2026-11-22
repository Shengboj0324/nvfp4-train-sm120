#!/usr/bin/env bash
# Main convergence experiment: GPT-2-small-sized model, TinyStories, 3000 steps x 16k tokens.
set -uo pipefail
cd "$(dirname "$0")/.."
R=experiments/results
mkdir -p $R
run() { name=$1; shift
  tools/run.sh train_$name experiments/train_gpt.py --steps 3000 --out $R/$name.jsonl "$@" > $R/$name.stdout 2>&1
  echo "$name exit=$? $(tail -1 $R/$name.stdout)"; sleep 60; }
run bf16          --precision bf16
run nvfp4         --precision nvfp4 --hp-last 2
run nvfp4_all     --precision nvfp4 --hp-last 0
run nvfp4_no_rht  --precision nvfp4 --hp-last 2 --no-rht
cp logs/train_*.csv $R/ 2>/dev/null || true
echo ALL_DONE
