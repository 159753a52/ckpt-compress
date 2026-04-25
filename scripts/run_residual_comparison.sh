#!/bin/bash
# Quick residual comparison: ExCP-residual vs Ours-3d-residual
# Only 3 methods: none (baseline), excp-residual, ours-3d-residual
# Target: <20 min total

set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ckpt
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE=/root/ckpt-compress/data/hf_cache
export ALPACA_LOCAL_PATH=/root/ckpt-compress/data/alpaca/alpaca_data.json

cd /root/ckpt-compress/code/checkpoint_compress

OUT=results/paper_results/residual_comparison_pythia_k5_r070
mkdir -p $OUT

echo "============================================================"
echo "Residual Pruning: ExCP vs Ours-3d (both on delta, uniform)"
echo "Pythia-410M / Alpaca | K=5 | ratio=0.70 | lr=5e-4"
echo "============================================================"
echo "Start: $(date)"

python experiments/scripts/run_fault_tolerant_training.py \
  --model pythia-410m --dataset alpaca \
  --checkpoint checkpoints/pythia_410m_alpaca_1000steps/checkpoint_step_1000.pt \
  --total_steps 800 --num_recoveries 5 \
  --lr 5e-4 --lr_schedule cosine \
  --batch_size 2 --seq_length 128 \
  --methods none,excp-residual+uniform,ours-3d-residual+uniform \
  --num_importance_steps 10 --eval_interval 50 \
  --device cuda \
  --prune_ratio 0.70 \
  --seed 42 \
  --output_dir $OUT 2>&1 | tee $OUT/run.log

echo ""
echo "End: $(date)"
echo "============================================================"
echo "Results in: $OUT"
ls -la $OUT/
