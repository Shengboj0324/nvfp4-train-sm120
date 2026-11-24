#!/usr/bin/env bash
# v0.10 recipe ablations on GPT-124M, three seeds each (experiments/run_ablation.sh), after baseline_sr8
# (the v0.9 recipe with the 8-bit integer stochastic rounder).
set -uo pipefail
cd "$(dirname "$0")/.."
bash experiments/run_ablation.sh four_six --four-six
bash experiments/run_ablation.sh ms_eden --ms-eden
bash experiments/run_ablation.sh ms_eden_four_six --ms-eden --four-six
bash experiments/run_ablation.sh fc2_bf16 --hp-modules fc2
echo ALL_ABLATIONS_DONE
