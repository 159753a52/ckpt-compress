#!/bin/bash
# 完整实验脚本：所有表格数据
# Table 3: GPT-2 Medium K=10 主实验
# Table 4: 消融实验
# Table 5: 剪枝+量化协同
# BERT-Large SST-2 K=10 补充

set -e
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

GPT2_CKPT="checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt"
BERT_CKPT="checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000_final.pt"

mkdir -p logs results/table3 results/table4 results/table5 results/bert_supp

echo '================================================'
echo 'Phase 1: Table 3 - GPT-2 Medium K=10 Main Sweep'
echo '================================================'

# Methods: none, magnitude+uniform, first-order+uniform (Inshrinkerator proxy),
#          residual-magnitude+uniform (ExCP), ours-2d
METHODS="none,magnitude+uniform,first-order+uniform,residual-magnitude+uniform,ours-2d"

for RATIO in 0.1 0.2 0.3 0.4; do
    echo ""
    echo "=== Table3: GPT-2 K=10 ratio=${RATIO} ==="
    OUTDIR="results/table3/gpt2m_k10_p${RATIO}"
    mkdir -p "$OUTDIR"

    python experiments/scripts/run_fault_tolerant_training.py \
        --model gpt2-medium \
        --dataset wikitext103 \
        --checkpoint $GPT2_CKPT \
        --total_steps 1000 \
        --num_recoveries 10 \
        --lr 5e-5 \
        --lr_schedule cosine \
        --batch_size 4 \
        --seq_length 512 \
        --methods $METHODS \
        --num_importance_steps 10 \
        --eval_interval 50 \
        --device cuda \
        --prune_ratio $RATIO \
        --output_dir "$OUTDIR"

    echo "Done Table3 ratio=${RATIO}!"
done

echo '================================================'
echo 'Phase 2: Table 4 - Ablation on GPT-2 Medium'
echo '================================================'

# Ablation: test component contributions at p=0.2 and p=0.4
# magnitude+uniform (magnitude only), first-order+uniform (FO only),
# magnitude+weibull-adaptive (alloc only), ours-2d (full)
ABLATION_METHODS="magnitude+uniform,first-order+uniform,magnitude+weibull-adaptive,ours-2d"

for RATIO in 0.2 0.4; do
    echo ""
    echo "=== Table4: Ablation ratio=${RATIO} ==="
    OUTDIR="results/table4/ablation_p${RATIO}"
    mkdir -p "$OUTDIR"

    python experiments/scripts/run_fault_tolerant_training.py \
        --model gpt2-medium \
        --dataset wikitext103 \
        --checkpoint $GPT2_CKPT \
        --total_steps 1000 \
        --num_recoveries 10 \
        --lr 5e-5 \
        --lr_schedule cosine \
        --batch_size 4 \
        --seq_length 512 \
        --methods $ABLATION_METHODS \
        --num_importance_steps 10 \
        --eval_interval 50 \
        --device cuda \
        --prune_ratio $RATIO \
        --output_dir "$OUTDIR"

    echo "Done Table4 ablation ratio=${RATIO}!"
done

echo '================================================'
echo 'Phase 3: Table 5 - Pruning + Quantization Synergy'
echo '================================================'

# Synergy: compare with/without quantization at p=0.3 and p=0.4
SYNERGY_METHODS="magnitude+uniform,magnitude+uniform+kmeans256,ours-2d,ours-2d+kmeans256"

for RATIO in 0.3 0.4; do
    echo ""
    echo "=== Table5: Synergy ratio=${RATIO} ==="
    OUTDIR="results/table5/synergy_p${RATIO}"
    mkdir -p "$OUTDIR"

    python experiments/scripts/run_fault_tolerant_training.py \
        --model gpt2-medium \
        --dataset wikitext103 \
        --checkpoint $GPT2_CKPT \
        --total_steps 1000 \
        --num_recoveries 10 \
        --lr 5e-5 \
        --lr_schedule cosine \
        --batch_size 4 \
        --seq_length 512 \
        --methods $SYNERGY_METHODS \
        --num_importance_steps 10 \
        --eval_interval 50 \
        --device cuda \
        --prune_ratio $RATIO \
        --output_dir "$OUTDIR"

    echo "Done Table5 synergy ratio=${RATIO}!"
done

echo '================================================'
echo 'Phase 4: BERT-Large SST-2 K=10 Supplementary'
echo '================================================'

BERT_METHODS="none,magnitude+uniform,ours-2d"

for RATIO in 0.1 0.2 0.3 0.4; do
    echo ""
    echo "=== BERT SST-2 K=10 ratio=${RATIO} ==="
    OUTDIR="results/bert_supp/bert_sst2_k10_p${RATIO}"
    mkdir -p "$OUTDIR"

    python experiments/scripts/run_fault_tolerant_training.py \
        --model bert-large \
        --dataset sst2 \
        --checkpoint $BERT_CKPT \
        --total_steps 1000 \
        --num_recoveries 10 \
        --lr 1e-5 \
        --lr_schedule cosine \
        --batch_size 4 \
        --seq_length 128 \
        --methods $BERT_METHODS \
        --num_importance_steps 10 \
        --eval_interval 50 \
        --device cuda \
        --prune_ratio $RATIO \
        --output_dir "$OUTDIR"

    echo "Done BERT SST-2 ratio=${RATIO}!"
done

echo '================================================'
echo 'ALL EXPERIMENTS COMPLETE!'
echo '================================================'
date
