#!/bin/bash
# BERT-Large MNLI 稀疏率扫描 + GPT-2 Medium 低lr实验
# 为论文提供更多真实数据

set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh && conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

echo "=========================================="
echo "Part 1: BERT-Large MNLI Sparsity Sweep"
echo "=========================================="

MNLI_CKPT="checkpoints/bert_large_mnli_1000steps/checkpoint_step_1000_final.pt"

# MNLI oracle
python experiments/scripts/run_fault_tolerant_training.py \
  --model bert-large --dataset mnli \
  --checkpoint ${MNLI_CKPT} \
  --total_steps 1000 --num_recoveries 5 \
  --lr 1e-5 --lr_schedule cosine \
  --batch_size 4 --seq_length 128 \
  --methods none \
  --num_importance_steps 10 --eval_interval 20 \
  --device cuda \
  --output_dir results/paper_results/bert_mnli_sweep_oracle

# MNLI sweep (only 0.2 and 0.4 for key data points)
for RATIO in 0.2 0.4; do
  echo ">>> MNLI Prune ratio = ${RATIO} <<<"
  python experiments/scripts/run_fault_tolerant_training.py \
    --model bert-large --dataset mnli \
    --checkpoint ${MNLI_CKPT} \
    --total_steps 1000 --num_recoveries 5 \
    --lr 1e-5 --lr_schedule cosine \
    --batch_size 4 --seq_length 128 \
    --methods magnitude+uniform,first-order+uniform,magnitude+weibull-adaptive \
    --num_importance_steps 10 --eval_interval 20 \
    --device cuda \
    --prune_ratio ${RATIO} \
    --output_dir results/paper_results/bert_mnli_sweep_${RATIO}
done

echo "=========================================="
echo "Part 2: GPT-2 Medium Low LR Experiment"  
echo "=========================================="

GPT2_CKPT="checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt"

# 用极低 lr 保持 PPL 不退化
for RATIO in 0.2 0.4; do
  echo ">>> GPT-2M Prune ratio = ${RATIO} <<<"
  python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium --dataset wikitext103 \
    --checkpoint ${GPT2_CKPT} \
    --total_steps 1000 --num_recoveries 5 \
    --lr 2e-6 --lr_schedule cosine \
    --batch_size 2 --seq_length 128 \
    --methods none,magnitude+uniform,first-order+uniform,magnitude+weibull-adaptive \
    --num_importance_steps 10 --eval_interval 20 \
    --device cuda \
    --prune_ratio ${RATIO} \
    --output_dir results/paper_results/gpt2m_sweep_lr2e6_${RATIO}
done

echo "=========================================="
echo "ALL EXPERIMENTS DONE"
echo "=========================================="
