#!/bin/bash
# 轻量实验：BERT-Large / MNLI
# 200步, K=2, prune_ratio=0.3, 3条曲线
# 预计耗时 ~50 分钟

set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh && conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

echo "=== BERT-Large MNLI Lightweight FT Experiment ==="
echo "Steps=200, K=2, ratio=0.3, cosine LR"
echo "Start: $(date)"

python -u experiments/scripts/run_fault_tolerant_training.py \
    --model bert-large \
    --dataset mnli \
    --checkpoint checkpoints/bert_large_mnli_1000steps/checkpoint_step_1000_final.pt \
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
    --data_dir /root/data \
    --output_dir results/paper_results/lightweight_bert_mnli_K2_p03

echo "=== Done: $(date) ==="
echo "Results in: results/paper_results/lightweight_bert_mnli_K2_p03/"
