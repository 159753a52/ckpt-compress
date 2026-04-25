#!/bin/bash
# Quick test: verify fixed ours-2d beats magnitude+uniform
# Run AFTER batch2 finishes

set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh && conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

echo "=========================================="
echo "Quick Test: ours-2d vs magnitude (BERT-Large SST-2)"
echo "=========================================="

# K=5, p=0.3, cosine LR — same setting where magnitude+weibull was 93.00% vs uniform 92.32%
python experiments/scripts/run_fault_tolerant_training.py \
  --model bert-large --dataset sst2 \
  --checkpoint checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000_final.pt \
  --total_steps 1000 --num_recoveries 5 \
  --lr 1e-5 --lr_schedule cosine \
  --batch_size 4 --seq_length 128 \
  --methods none,magnitude+uniform,ours-2d \
  --num_importance_steps 10 --eval_interval 20 \
  --device cuda \
  --prune_ratio 0.3 \
  --output_dir results/paper_results/ours2d_test

echo "=========================================="
echo "Quick Test: ours-2d vs magnitude (GPT-2 Medium)"
echo "=========================================="

# K=5, p=0.4, cosine LR — same setting where weibull was 25.76 vs uniform 26.17
python experiments/scripts/run_fault_tolerant_training.py \
  --model gpt2-medium --dataset wikitext103 \
  --checkpoint checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt \
  --total_steps 1000 --num_recoveries 5 \
  --lr 5e-5 --lr_schedule cosine \
  --batch_size 4 --seq_length 512 \
  --methods none,magnitude+uniform,ours-2d \
  --num_importance_steps 10 --eval_interval 20 \
  --device cuda \
  --prune_ratio 0.4 \
  --output_dir results/paper_results/ours2d_test_gpt2

echo "=========================================="
echo "ALL TESTS DONE"
echo "=========================================="
