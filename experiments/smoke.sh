#!/usr/bin/env bash
# 60-step smoke test of both precisions (speed, memory, finite loss).
set -euo pipefail
cd "$(dirname "$0")/.."
for p in bf16 nvfp4; do
  tools/run.sh smoke_$p experiments/train_gpt.py --precision $p --steps 60 --eval-every 1000 --out logs/smoke_$p.jsonl 2>&1 | grep -E "nvfp4_linears|'step': (50|0)|guard|Error|error" || true
done
