#!/bin/bash
# Batch 3: BERT STS-B + GPT-2 Small experiments
# Purpose: Additional real data points for paper supplementary / validation

set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh && conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

echo "=========================================="
echo "Part 1: BERT-Large STS-B Sweep"
echo "=========================================="

STSB_CKPT="checkpoints/bert_large_stsb_1000steps/checkpoint_step_1000_final.pt"

# STS-B oracle
python experiments/scripts/run_fault_tolerant_training.py \
  --model bert-large --dataset stsb \
  --checkpoint ${STSB_CKPT} \
  --total_steps 1000 --num_recoveries 5 \
  --lr 1e-5 --lr_schedule cosine \
  --batch_size 4 --seq_length 128 \
  --methods none \
  --num_importance_steps 10 --eval_interval 20 \
  --device cuda \
  --output_dir results/paper_results/bert_stsb_sweep_oracle

# STS-B sweep
for RATIO in 0.2 0.4; do
  echo ">>> STS-B Prune ratio = ${RATIO} <<<"
  python experiments/scripts/run_fault_tolerant_training.py \
    --model bert-large --dataset stsb \
    --checkpoint ${STSB_CKPT} \
    --total_steps 1000 --num_recoveries 5 \
    --lr 1e-5 --lr_schedule cosine \
    --batch_size 4 --seq_length 128 \
    --methods magnitude+uniform,first-order+uniform,magnitude+weibull-adaptive \
    --num_importance_steps 10 --eval_interval 20 \
    --device cuda \
    --prune_ratio ${RATIO} \
    --output_dir results/paper_results/bert_stsb_sweep_${RATIO}
done

echo "=========================================="
echo "Part 2: GPT-2 Small WikiText-103 Sweep"
echo "=========================================="

GPT2S_CKPT="checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt"

# GPT-2 Small oracle 
python experiments/scripts/run_fault_tolerant_training.py \
  --model gpt2-small --dataset wikitext103 \
  --checkpoint ${GPT2S_CKPT} \
  --total_steps 1000 --num_recoveries 5 \
  --lr 5e-5 --lr_schedule cosine \
  --batch_size 4 --seq_length 128 \
  --methods none \
  --num_importance_steps 10 --eval_interval 20 \
  --device cuda \
  --output_dir results/paper_results/gpt2s_sweep_oracle

# GPT-2 Small sweep
for RATIO in 0.2 0.4; do
  echo ">>> GPT-2 Small Prune ratio = ${RATIO} <<<"
  python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-small --dataset wikitext103 \
    --checkpoint ${GPT2S_CKPT} \
    --total_steps 1000 --num_recoveries 5 \
    --lr 5e-5 --lr_schedule cosine \
    --batch_size 4 --seq_length 128 \
    --methods magnitude+uniform,first-order+uniform,magnitude+weibull-adaptive \
    --num_importance_steps 10 --eval_interval 20 \
    --device cuda \
    --prune_ratio ${RATIO} \
    --output_dir results/paper_results/gpt2s_sweep_${RATIO}
done

echo "=========================================="
echo "ALL BATCH3 EXPERIMENTS DONE"
echo "=========================================="
