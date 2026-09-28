#!/usr/bin/env bash
# Create checkpoints/pythia_410m_alpaca/checkpoint_step_1000.pt on GPU 7 with
# the repository fine-tuning script and its default hyper-parameters
# (1000 steps, batch 4, sequence 512, lr 2e-5, AMP). Only the final step is saved.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/gpu7_env.sh"
out="$DACP_ASSETS/checkpoints/pythia_410m_alpaca"
mkdir -p "$out"
cd "$DACP_CODE"
"$PYTHON" experiments/scripts/finetune/finetune_pythia_410m.py \
  --total_steps 1000 --batch_size 4 --seq_length 512 --lr 2e-5 \
  --save_every 1000 --device cuda --output_dir "$out"
