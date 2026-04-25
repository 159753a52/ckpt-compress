#!/bin/bash
# ============================================================
# Table 3: Pruning–Quantization Synergy (GPT-2 Medium K=10)
#
# 与 K=10 sweep 完全一致的配置:
#   K=10, lr=5e-5, cosine, batch=4, seq=512
#
# 只跑 Prune+Quant 组合（Prune-only 数据来自 sweep）:
#   magnitude+uniform+kmeans256 @ p=0.3, 0.4
#   ours-2d+kmeans256 @ p=0.3, 0.4
#
# 预计: ~3-4 小时（4 configs × K=10 × ~2min/seg）
# ============================================================
set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

CKPT="checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt"

echo "========================================="
echo "Table 3 Synergy: GPT-2 Medium K=10"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

for PRUNE in 0.3 0.4; do
    echo ""
    echo ">>> Prune+Quant @ p=${PRUNE} <<<"
    
    python experiments/scripts/run_fault_tolerant_training.py \
        --model gpt2-medium --dataset wikitext103 \
        --checkpoint "$CKPT" \
        --total_steps 1000 --num_recoveries 10 \
        --prune_ratio "$PRUNE" \
        --seq_length 512 --batch_size 4 \
        --lr 5e-5 --lr_schedule cosine \
        --methods "magnitude+uniform+kmeans256,ours-2d+kmeans256" \
        --eval_interval 10 --num_importance_steps 10 \
        --alpha 0.7 --protection_ratio 0.001 \
        --device cuda \
        --output_dir "results/paper_results/gpt2m_k10_synergy_p${PRUNE}"
    
    echo "  p=${PRUNE} done: $(date '+%H:%M:%S')"
done

echo ""
echo "========================================="
echo "Synergy 实验完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
