#!/bin/bash
# GPT-2 Medium K=10 Ablation — Table 2 验证
# 5种变体 × 2种稀疏率（0.2, 0.4）
# 运行：nohup bash scripts/run_gpt2_k10_ablation.sh > logs/gpt2_k10_ablation.log 2>&1 &

set -e
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

CHECKPOINT="checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt"
MODEL="gpt2-medium"
DATASET="wikitext103"
TOTAL_STEPS=1000
K=10
LR="5e-5"
BATCH=4
SEQ=512
EVAL_INT=20
IMP_STEPS=10

mkdir -p logs

echo "=========================================="
echo "GPT-2 Medium K=10 Ablation — Table 2"
echo "=========================================="

# Table 2 变体:
# 1. Magnitude + Uniform (已在 sweep 中)
# 2. First-order + Uniform
# 3. First+Second (ours-2d) + Uniform
# 4. First-order + Dist-aware (weibull)
# 5. First+Second (ours-2d) + Dist-aware (已在 sweep 中)
# 所以只需补充 2,3,4

for RATIO in 0.2 0.4; do
    echo ""
    echo ">>> Ablation ratio=${RATIO} <<<"

    OUTDIR="results/paper_results/gpt2m_k10_ablation_${RATIO}"
    mkdir -p "$OUTDIR"

    python experiments/scripts/run_fault_tolerant_training.py \
        --model $MODEL \
        --dataset $DATASET \
        --checkpoint $CHECKPOINT \
        --total_steps $TOTAL_STEPS \
        --num_recoveries $K \
        --lr $LR \
        --lr_schedule cosine \
        --batch_size $BATCH \
        --seq_length $SEQ \
        --methods first-order+uniform,first-order+weibull-adaptive \
        --num_importance_steps $IMP_STEPS \
        --eval_interval $EVAL_INT \
        --device cuda \
        --prune_ratio $RATIO \
        --output_dir "$OUTDIR"

    echo "Done ablation ratio=${RATIO}!"
done

echo "=========================================="
echo "All ablation experiments done!"
echo "=========================================="
