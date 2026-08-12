#!/bin/bash
# Quick validation: ours-3d vs ours-2d vs magnitude on Pythia K=1, ratio=0.2
# Purpose: verify ours-3d (with real second-order info) performs better

set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ckpt
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE=/root/ckpt-compress/data/hf_cache
export ALPACA_LOCAL_PATH=/root/ckpt-compress/data/alpaca/alpaca_data.json

cd /root/ckpt-compress/code/checkpoint_compress

OUT=results/paper_results/verify_ours3d_pythia_k1_r020
mkdir -p $OUT

echo "============================================================"
echo "Pythia-410M / Alpaca | K=1 | ratio=0.20"
echo "Methods: magnitude+uniform, ours-2d+uniform, ours-3d+uniform, ours-3d (weibull default)"
echo "============================================================"
echo "Start: $(date)"

python experiments/scripts/run_fault_tolerant_training.py \
  --model pythia-410m --dataset alpaca \
  --checkpoint checkpoints/pythia_410m_alpaca_1000steps/checkpoint_step_1000.pt \
  --total_steps 100 --num_recoveries 1 \
  --lr 2e-5 --lr_schedule cosine \
  --batch_size 2 --seq_length 128 \
  --methods none,magnitude+uniform,ours-2d+uniform,ours-3d+uniform,ours-3d \
  --num_importance_steps 10 --eval_interval 50 \
  --device cuda \
  --prune_ratio 0.20 \
  --seed 42 \
  --output_dir $OUT 2>&1 | tee $OUT/run.log

echo "End: $(date)"
echo "============================================================"
echo "Results in: $OUT"
ls -la $OUT/
