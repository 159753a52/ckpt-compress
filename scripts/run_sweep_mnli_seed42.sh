#!/bin/bash
set -e
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh && conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress
echo "=== BERT-Large MNLI Seed=42 Sweep ==="
echo "Start: $(date)"
for RATIO in 0.1 0.2 0.3 0.4; do
    echo ""
    echo "=========================================="
    echo "  prune_ratio=${RATIO}, seed=42"
    echo "  Time: $(date)"
    echo "=========================================="
    python -u experiments/scripts/run_fault_tolerant_training.py \
        --model bert-large --dataset mnli \
        --checkpoint checkpoints/bert_large_mnli_1000steps/checkpoint_step_1000_final.pt \
        --total_steps 200 --num_recoveries 2 --prune_ratio ${RATIO} \
        --methods none,magnitude+uniform,ours-2d \
        --lr 1e-5 --lr_schedule cosine --batch_size 4 --seq_length 128 \
        --num_importance_steps 10 --eval_interval 10 --device cuda \
        --data_dir /root/data --seed 42
done
echo ""
echo "=== All ratios done: $(date) ==="