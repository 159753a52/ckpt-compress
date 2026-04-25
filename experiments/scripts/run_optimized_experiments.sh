#!/bin/bash
# =============================================================================
# Phase 1: 快速验证 —— GPT-2M 单模型测试优化效果
#
# 改进：
#   1. 多 batch 梯度累积（importance.py 补丁，自动用所有 cached batch）
#   2. normalize=True（一阶/二阶项归一化）
#   3. hvp_batches=4（平衡速度与精度）
#   4. alpha 扫描 3 个值 (0.1, 0.5, 1.0)
#   5. 高稀疏率扩展到 0.5, 0.6
#
# 用法:
#   nohup bash experiments/scripts/run_optimized_experiments.sh \
#     > /root/exp_logs/optimized.log 2>&1 &
# =============================================================================
set -e

export PYTHONUNBUFFERED=1
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate GPT-2
source experiments/scripts/setup_env.sh
export HF_HUB_DISABLE_DISK_SPACE_CHECK=1

mkdir -p /root/exp_logs

echo "========================================="
echo " Phase 1: 快速验证批次 - $(date)"
echo "========================================="

# --------------------------------------------------------------------------
# Step 0: 应用 importance.py 补丁（多batch梯度累积）
# --------------------------------------------------------------------------
echo ""
echo "[Step 0] 应用补丁..."
python experiments/scripts/patch_importance.py
python experiments/scripts/patch_methods.py
python experiments/scripts/patch_scoring.py
echo "[Step 0] 补丁完成"

# ==========================================================================
# [1/3]  GPT-2M  Table 3 主结果 — 优化后 alpha 扫描
# ==========================================================================
echo ""
echo "[1/3] GPT-2M 主结果: normalize + hvp_batches=4 + alpha扫描  $(date)"
python experiments/scripts/run_method_comparison.py \
    --model gpt2-medium --dataset wikitext103 \
    --checkpoint /root/checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4,0.5,0.6 \
    --alpha 0.1,0.5,1.0 \
    --num_steps 50 --eval_batches 50 --hvp_batches 4 \
    --seq_length 128 --batch_size 2 \
    --hvp_mode block --normalize \
    --device cuda \
    --output_dir results/paper_results/table1_optimized
echo "[1/3] 完成 $(date)"

# ==========================================================================
# [2/3]  Pythia  Table 3 主结果 — 优化后 alpha 扫描
# ==========================================================================
echo ""
echo "[2/3] Pythia-410M 主结果: normalize + hvp_batches=4 + alpha扫描  $(date)"
python experiments/scripts/run_method_comparison.py \
    --model pythia-410m --dataset alpaca \
    --checkpoint /root/checkpoints/pythia_410m_alpaca/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4,0.5,0.6 \
    --alpha 0.1,0.5,1.0 \
    --num_steps 50 --eval_batches 50 --hvp_batches 4 \
    --seq_length 128 --batch_size 2 \
    --hvp_mode block --normalize \
    --device cuda \
    --output_dir results/paper_results/table1_optimized
echo "[2/3] 完成 $(date)"

# ==========================================================================
# [3/3]  ViT  Table 3 主结果 — 优化后 alpha 扫描
# ==========================================================================
echo ""
echo "[3/3] ViT-L/32 主结果: normalize + hvp_batches=4 + alpha扫描  $(date)"
IMAGE_SIZE=384 python experiments/scripts/run_method_comparison.py \
    --model vit-l-32 --dataset imagenet \
    --checkpoint /root/checkpoints/vit_l32_imagenet/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4,0.5,0.6 \
    --alpha 0.1,0.5,1.0 \
    --num_steps 50 --eval_batches 50 --hvp_batches 4 \
    --batch_size 16 \
    --hvp_mode block --normalize \
    --device cuda \
    --output_dir results/paper_results/table1_optimized
echo "[3/3] 完成 $(date)"

echo ""
echo "========================================="
echo " Phase 1 完成 - $(date)"
echo "========================================="
