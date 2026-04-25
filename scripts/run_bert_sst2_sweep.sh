#!/bin/bash
# BERT-Large SST-2 稀疏率扫描 (K=5, cosine)
# 为 Table 1 提供真实数据点

set -e

source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh && conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

CKPT="checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000_final.pt"

COMMON_ARGS="--model bert-large --dataset sst2 \
  --checkpoint ${CKPT} \
  --total_steps 1000 --num_recoveries 5 \
  --lr 1e-5 --lr_schedule cosine \
  --batch_size 4 --seq_length 128 \
  --methods none,magnitude+uniform,first-order+uniform,magnitude+weibull-adaptive \
  --num_importance_steps 10 --eval_interval 20 \
  --device cuda"

echo "=========================================="
echo "BERT-Large SST-2 Sparsity Sweep (K=5, cosine)"
echo "=========================================="

# 先跑 none 作为 oracle baseline（只需一次）
echo ">>> Running oracle baseline (none) <<<"
python experiments/scripts/run_fault_tolerant_training.py \
  ${COMMON_ARGS} --prune_ratio 0.3 \
  --methods none \
  --output_dir results/paper_results/bert_sst2_sweep_oracle

# 再跑不同稀疏率下的压缩方法
for RATIO in 0.1 0.2 0.3 0.4; do
  echo ""
  echo ">>> Prune ratio = ${RATIO} <<<"
  echo ""
  python experiments/scripts/run_fault_tolerant_training.py \
    ${COMMON_ARGS} --prune_ratio ${RATIO} \
    --methods magnitude+uniform,first-order+uniform,magnitude+weibull-adaptive \
    --output_dir results/paper_results/bert_sst2_sweep_${RATIO}
done

echo ""
echo "=========================================="
echo "ALL SWEEPS DONE"
echo "=========================================="
