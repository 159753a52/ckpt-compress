#!/bin/bash
# 轻量多稀疏率 sweep：GPT-2 Medium / WikiText-103
# K=2, 200步, 4个稀疏率
# 预计总耗时 ~90 分钟（每个稀疏率 ~22 分钟）

set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh && conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

echo "=== GPT-2 Medium Multi-Ratio Sweep ==="
echo "K=2, Steps=200, cosine LR"
echo "Start: $(date)"

for RATIO in 0.1 0.2 0.3 0.4; do
    echo ""
    echo "=========================================="
    echo "  Running prune_ratio=${RATIO}"
    echo "  Time: $(date)"
    echo "=========================================="
    
    python -u experiments/scripts/run_fault_tolerant_training.py \
        --model gpt2-medium \
        --dataset wikitext103 \
        --checkpoint checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt \
        --total_steps 200 \
        --num_recoveries 2 \
        --prune_ratio ${RATIO} \
        --methods none,magnitude+uniform,ours-2d \
        --lr 5e-5 \
        --lr_schedule cosine \
        --batch_size 4 \
        --seq_length 512 \
        --num_importance_steps 10 \
        --eval_interval 10 \
        --device cuda \
        --output_dir "results/paper_results/sweep_gpt2m_K2_p${RATIO}"
done

echo ""
echo "=== All ratios done: $(date) ==="
