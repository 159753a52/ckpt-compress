#!/usr/bin/env bash
# Create checkpoints/pythia_410m_alpaca_step3000/checkpoint_step_3000.pt on GPU 7
# with the repository fine-tuning script and the same hyper-parameters as the
# step-1000 checkpoint (batch 4, sequence 512, lr 2e-5, AMP). The script cannot
# resume, so this is an independent run with --total_steps 3000 (300 warmup
# steps + cosine decay over 3000 steps); only step 3000 is saved.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/gpu7_env.sh"
out="$DACP_ASSETS/checkpoints/pythia_410m_alpaca_step3000"
mkdir -p "$out"
cd "$DACP_CODE"
if [[ -f "$out/checkpoint_step_3000.pt" ]]; then
  echo "exists: $out/checkpoint_step_3000.pt"
  exit 0
fi
start=$(date +%s)
"$PYTHON" experiments/scripts/finetune/finetune_pythia_410m.py \
  --total_steps 3000 --batch_size 4 --seq_length 512 --lr 2e-5 \
  --save_every 3000 --device cuda --output_dir "$out"
echo "finetune finished after $(( $(date +%s) - start )) s"
