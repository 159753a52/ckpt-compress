#!/bin/bash
# batch4: Table 1/2/3 验证实验
# 1. GPT-2 Medium K=10 cosine (Table 1 核心验证)
# 2. Ablation study (Table 2 验证)
# 3. Joint compression / synergy (Table 3 验证)

set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh && conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

GPT2_CKPT="checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt"

echo "=========================================="
echo "Part 1: GPT-2 Medium K=10 Cosine (Table 1)"
echo "=========================================="

# K=10 oracle (cosine lr)
python experiments/scripts/run_fault_tolerant_training.py \
  --model gpt2-medium --dataset wikitext103 \
  --checkpoint ${GPT2_CKPT} \
  --total_steps 2000 --num_recoveries 10 \
  --lr 5e-5 --lr_schedule cosine \
  --batch_size 2 --seq_length 128 \
  --methods none \
  --num_importance_steps 10 --eval_interval 20 \
  --device cuda \
  --output_dir results/paper_results/gpt2m_k10_cosine_oracle

# K=10 sweep: sparsity 20% and 40% (two key data points for Table 1)
for RATIO in 0.2 0.4; do
  echo ">>> GPT-2M K=10 Cosine, Prune ratio = ${RATIO} <<<"
  python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium --dataset wikitext103 \
    --checkpoint ${GPT2_CKPT} \
    --total_steps 2000 --num_recoveries 10 \
    --lr 5e-5 --lr_schedule cosine \
    --batch_size 2 --seq_length 128 \
    --methods magnitude+uniform,magnitude+weibull-adaptive,second-order-hvp+weibull-adaptive \
    --num_importance_steps 10 --eval_interval 20 \
    --device cuda \
    --prune_ratio ${RATIO} \
    --output_dir results/paper_results/gpt2m_k10_cosine_${RATIO}
done

echo "=========================================="
echo "Part 2: Ablation Study on GPT-2 Medium (Table 2)"
echo "=========================================="

# Ablation: 6 combinations of importance x allocation
python experiments/scripts/run_ablation_study.py \
  --model gpt2-medium --dataset wikitext103 \
  --checkpoint ${GPT2_CKPT} \
  --prune_ratios 0.2,0.4 \
  --num_steps 100 --hvp_batches 8 --eval_batches 20 \
  --hvp_mode block --chunk_size 10 \
  --batch_size 2 --seq_length 128 \
  --device cuda \
  --output_dir results/paper_results/ablation_gpt2m

echo "=========================================="
echo "Part 3: FT Synergy (Table 3) - K=10 Prune+Quant"
echo "=========================================="

# K=10 FT with pruning-only vs pruning+kmeans256 at 30% and 40%
for RATIO in 0.3 0.4; do
  echo ">>> GPT-2M K=10 Synergy, Prune ratio = ${RATIO} <<<"
  python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium --dataset wikitext103 \
    --checkpoint ${GPT2_CKPT} \
    --total_steps 2000 --num_recoveries 10 \
    --lr 5e-5 --lr_schedule cosine \
    --batch_size 2 --seq_length 128 \
    --methods magnitude+uniform,magnitude+uniform+kmeans256,second-order-hvp+weibull-adaptive,second-order-hvp+weibull-adaptive+kmeans256 \
    --num_importance_steps 10 --eval_interval 20 \
    --device cuda \
    --prune_ratio ${RATIO} \
    --output_dir results/paper_results/gpt2m_k10_synergy_${RATIO}
done

echo "=========================================="
echo "ALL BATCH4 EXPERIMENTS DONE"
echo "=========================================="
