#!/bin/bash
# GPT-2 Medium K=10 Sweep — 多稀疏度，cosine LR
# 目的：获取 Table 1 GPT-2 列的真实对照数据
# 运行：GPU 空闲后 nohup bash scripts/run_gpt2_k10_sweep.sh > logs/gpt2_k10_sweep.log 2>&1 &

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

mkdir -p logs results/paper_results

echo "=========================================="
echo "GPT-2 Medium K=10 Sweep — Table 1 验证"
echo "=========================================="

for RATIO in 0.1 0.2 0.3 0.4; do
    echo ""
    echo ">>> GPT-2 K=10 Prune ratio = ${RATIO} <<<"

    OUTDIR="results/paper_results/gpt2m_k10_sweep_${RATIO}"
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
        --methods none,magnitude+uniform,magnitude+weibull-adaptive,ours-2d \
        --num_importance_steps $IMP_STEPS \
        --eval_interval $EVAL_INT \
        --device cuda \
        --prune_ratio $RATIO \
        --output_dir "$OUTDIR"

    echo "Done ratio=${RATIO}!"
    echo ""
done

echo "=========================================="
echo "All GPT-2 K=10 sweep done!"
echo "=========================================="
