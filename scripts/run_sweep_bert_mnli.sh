#!/bin/bash
# 轻量多稀疏率 sweep：BERT-Large / MNLI
# K=2, 200步, 4个稀疏率
# 预计总耗时 ~2 小时（每个稀疏率 ~30 分钟）

set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh && conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

echo "=== BERT-Large MNLI Multi-Ratio Sweep ==="
echo "K=2, Steps=200, cosine LR"
echo "Start: $(date)"

for RATIO in 0.1 0.2 0.3 0.4; do
    echo ""
    echo "=========================================="
    echo "  Running prune_ratio=${RATIO}"
    echo "  Time: $(date)"
    echo "=========================================="
    
    python -u experiments/scripts/run_fault_tolerant_training.py \
        --model bert-large \
        --dataset mnli \
        --checkpoint checkpoints/bert_large_mnli_1000steps/checkpoint_step_1000_final.pt \
        --total_steps 200 \
        --num_recoveries 2 \
        --prune_ratio ${RATIO} \
        --methods none,magnitude+uniform,ours-2d \
        --lr 1e-5 \
        --lr_schedule cosine \
        --batch_size 4 \
        --seq_length 128 \
        --num_importance_steps 10 \
        --eval_interval 10 \
        --device cuda \
        --data_dir /root/data \
        --output_dir "results/paper_results/sweep_bert_mnli_K2_p${RATIO}"
done

echo ""
echo "=== All ratios done: $(date) ==="
