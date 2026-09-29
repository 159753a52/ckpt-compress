#!/usr/bin/env bash
# Usage: finetune_small_gpu7.sh <model_dir> <checkpoint_dir> [extra finetune args...]
# Fine-tune the local snapshot $DACP_ASSETS/models/<model_dir> on Alpaca with the repository
# recipe of finetune_pythia_410m.py (1000 steps, batch 4, sequence 512, lr 2e-5, AMP) on GPU 7
# and save model + optimizer at step 1000 to $DACP_ASSETS/checkpoints/<checkpoint_dir>.
# Extra args are passed through, e.g. "--dataset wikitext2" or
# "--micro_batch_size 1 --no-adamw_foreach" (gradient accumulation for larger models).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/gpu7_env.sh"
model_dir="$DACP_ASSETS/models/$1"
out="$DACP_ASSETS/checkpoints/$2"
shift 2
mkdir -p "$out"
cd "$DACP_CODE"
PYTHIA_MODEL_PATH="$model_dir" "$PYTHON" experiments/scripts/finetune/finetune_pythia_410m.py \
  --total_steps 1000 --batch_size 4 --seq_length 512 --lr 2e-5 \
  --save_every 1000 --device cuda --output_dir "$out" "$@"
