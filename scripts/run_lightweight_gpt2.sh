#!/bin/bash
# 轻量实验：GPT-2 Medium / WikiText-103
# 200步, K=2, prune_ratio=0.3, 3条曲线
# 预计耗时 3-5 分钟

set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh && conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

echo "=== GPT-2 Medium Lightweight FT Experiment ==="
echo "Steps=200, K=2, ratio=0.3, cosine LR"
echo "Start: $(date)"

python -u experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium \
    --dataset wikitext103 \
    --checkpoint checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt \
    --total_steps 200 \
    --num_recoveries 2 \
    --prune_ratio 0.3 \
    --methods none,magnitude+uniform,ours-2d \
    --lr 5e-5 \
    --lr_schedule cosine \
    --batch_size 4 \
    --seq_length 512 \
    --num_importance_steps 10 \
    --eval_interval 10 \
    --device cuda \
    --output_dir results/paper_results/lightweight_gpt2m_K2_p03

echo "=== Done: $(date) ==="
echo "Results in: results/paper_results/lightweight_gpt2m_K2_p03/"
