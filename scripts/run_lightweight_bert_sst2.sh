#!/bin/bash
# 轻量实验：BERT-Large / SST-2
# 200步, K=2, prune_ratio=0.3, 3条曲线
# 预计耗时 ~10 分钟

set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh && conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

echo "=== BERT-Large SST-2 Lightweight FT Experiment ==="
echo "Steps=200, K=2, ratio=0.3, cosine LR"
echo "Start: $(date)"

python -u experiments/scripts/run_fault_tolerant_training.py \
    --model bert-large \
    --dataset sst2 \
    --checkpoint checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000_final.pt \
    --total_steps 200 \
    --num_recoveries 2 \
    --prune_ratio 0.3 \
    --methods none,magnitude+uniform,ours-2d \
    --lr 1e-5 \
    --lr_schedule cosine \
    --batch_size 4 \
    --seq_length 128 \
    --num_importance_steps 10 \
    --eval_interval 10 \
    --device cuda \
    --output_dir results/paper_results/lightweight_bert_sst2_K2_p03

echo "=== Done: $(date) ==="
echo "Results in: results/paper_results/lightweight_bert_sst2_K2_p03/"
