#!/bin/bash
# ============================================================
# BERT-Large SST-2 K=10 Sweep (补充 Table 1 数据)
#
# K=10, lr=1e-5, cosine, batch=4, seq=128
# Methods: none, magnitude+uniform, magnitude+weibull-adaptive, ours-2d
# Sparsity: 0.1, 0.2, 0.3, 0.4
#
# 预计: ~4-5 小时
# ============================================================
set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

CKPT="checkpoints/bert_large_sst2/checkpoint_step_1000_final.pt"
METHODS="none,magnitude+uniform,magnitude+weibull-adaptive,ours-2d"

echo "========================================="
echo "BERT-Large SST-2 K=10 Sweep"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

for PRUNE in 0.1 0.2 0.3 0.4; do
    echo ""
    echo ">>> BERT SST-2 K=10 p=${PRUNE} <<<"
    
    python experiments/scripts/run_fault_tolerant_training.py \
        --model bert-large --dataset sst2 \
        --checkpoint "$CKPT" \
        --total_steps 1000 --num_recoveries 10 \
        --prune_ratio "$PRUNE" \
        --seq_length 128 --batch_size 4 \
        --lr 1e-5 --lr_schedule cosine \
        --methods "$METHODS" \
        --eval_interval 20 --num_importance_steps 10 \
        --alpha 0.7 --protection_ratio 0.001 \
        --device cuda \
        --output_dir "results/paper_results/bert_sst2_k10_sweep"
    
    echo "  p=${PRUNE} done: $(date '+%H:%M:%S')"
done

echo ""
echo "========================================="
echo "BERT SST-2 K=10 Sweep 完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
