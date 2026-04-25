#!/bin/bash
# 批量实验 A1-A4：分布拟合 + Block-wise HVP + 容错训练
# 预计总耗时 ~5h
# 用法：nohup bash experiments/scripts/run_batch_A.sh > /root/exp_logs/batch_A.log 2>&1 &

set -e
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate GPT-2
source experiments/scripts/setup_env.sh 2>/dev/null || true
export PYTHONUNBUFFERED=1

echo "========================================"
echo "批量实验 A：$(date)"
echo "========================================"

mkdir -p /root/exp_logs

# ===== A1: Pythia-410M 分布拟合验证 =====
echo ""
echo "========== A1: Pythia-410M 分布拟合 =========="
echo "开始时间: $(date)"

python experiments/scripts/run_gamma_validation.py \
    --model pythia-410m --dataset alpaca \
    --checkpoint /root/checkpoints/pythia_410m_alpaca/checkpoint_step_1000.pt \
    --prune_ratios 0.1,0.2,0.3,0.4,0.5,0.6 \
    --num_steps 50 --eval_batches 50 --hvp_batches 2 \
    --seq_length 128 --batch_size 2 --hvp_mode block \
    --data_dir /root/data --device cuda

echo "A1 完成: $(date)"

# ===== A2: Block-wise HVP 精度分析 =====
echo ""
echo "========== A2: Block-wise HVP 精度 =========="
echo "开始时间: $(date)"

# α=0.05, warmup=100 — 冲 Spearman 0.93+
python experiments/scripts/run_blockwise_hvp_analysis.py \
    --model gpt2-medium --dataset wikitext103 \
    --seq_length 128 --batch_size 2 \
    --hvp_batches 2 \
    --alpha_sweep 0.01,0.05,0.1 \
    --warmup_steps 100 --warmup_lr 1e-2 \
    --device cuda

echo "A2 完成: $(date)"

# ===== A3: 容错训练 GPT-2M =====
echo ""
echo "========== A3: 容错训练 GPT-2M =========="
echo "开始时间: $(date)"

python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium --dataset wikitext103 \
    --checkpoint /root/checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt \
    --total_steps 1000 --num_recoveries 5 \
    --prune_ratio 0.4 \
    --num_importance_steps 30 --eval_interval 10 \
    --batch_size 2 --seq_length 128 --lr 5e-5 \
    --data_dir /root/data --device cuda

echo "A3 完成: $(date)"

# ===== A4: 容错训练 BERT-Large =====
echo ""
echo "========== A4: 容错训练 BERT-Large =========="
echo "开始时间: $(date)"

python experiments/scripts/run_fault_tolerant_training.py \
    --model bert-large --dataset sst2 \
    --checkpoint /root/checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000.pt \
    --total_steps 1000 --num_recoveries 5 \
    --prune_ratio 0.4 \
    --num_importance_steps 30 --eval_interval 10 \
    --batch_size 8 --seq_length 128 --lr 2e-5 \
    --data_dir /root/data --device cuda

echo "A4 完成: $(date)"

echo ""
echo "========================================"
echo "全部完成: $(date)"
echo "========================================"
