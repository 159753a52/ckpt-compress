#!/bin/bash
# =============================================================================
# Table 3 (主结果) + Table 4 (消融) — 三个新模型
# GPT-2 Medium / WikiText-103, Pythia-410M / Alpaca, ViT-L/32 / ImageNet
#
# 用法: nohup bash experiments/scripts/run_new_models_experiments.sh \
#         > /root/exp_logs/new_models.log 2>&1 &
# =============================================================================
set -e

# ---------- 环境 ----------
export PYTHONUNBUFFERED=1
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate GPT-2
source experiments/scripts/setup_env.sh

# 额外磁盘空间检查
export HF_HUB_DISABLE_DISK_SPACE_CHECK=1

mkdir -p /root/exp_logs

echo "========================================="
echo " 新模型实验批次启动 - $(date)"
echo "========================================="

# ===========================================================================
# [1/6]  Table 4 消融  —  GPT-2 Medium + WikiText-103
# ===========================================================================
echo ""
echo "[1/6] Table 4 消融: GPT-2 Medium WikiText-103  $(date)"
python experiments/scripts/run_ablation_study.py \
    --model gpt2-medium --dataset wikitext103 \
    --checkpoint /root/checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 50 \
    --hvp_batches 2 --seq_length 128 --batch_size 2 \
    --hvp_mode block --device cuda
echo "[1/6] 完成 $(date)"

# ===========================================================================
# [2/6]  Table 3 主结果  —  GPT-2 Medium + WikiText-103
# ===========================================================================
echo ""
echo "[2/6] Table 3 主结果: GPT-2 Medium WikiText-103  $(date)"
python experiments/scripts/run_method_comparison.py \
    --model gpt2-medium --dataset wikitext103 \
    --checkpoint /root/checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 50 \
    --hvp_batches 2 --seq_length 128 --batch_size 2 \
    --hvp_mode block --device cuda
echo "[2/6] 完成 $(date)"

# ===========================================================================
# [3/6]  Table 4 消融  —  Pythia-410M + Alpaca
# ===========================================================================
echo ""
echo "[3/6] Table 4 消融: Pythia-410M Alpaca  $(date)"
python experiments/scripts/run_ablation_study.py \
    --model pythia-410m --dataset alpaca \
    --checkpoint /root/checkpoints/pythia_410m_alpaca/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 50 \
    --hvp_batches 2 --seq_length 128 --batch_size 2 \
    --hvp_mode block --device cuda
echo "[3/6] 完成 $(date)"

# ===========================================================================
# [4/6]  Table 3 主结果  —  Pythia-410M + Alpaca
# ===========================================================================
echo ""
echo "[4/6] Table 3 主结果: Pythia-410M Alpaca  $(date)"
python experiments/scripts/run_method_comparison.py \
    --model pythia-410m --dataset alpaca \
    --checkpoint /root/checkpoints/pythia_410m_alpaca/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 50 \
    --hvp_batches 2 --seq_length 128 --batch_size 2 \
    --hvp_mode block --device cuda
echo "[4/6] 完成 $(date)"

# ===========================================================================
# [5/6]  Table 4 消融  —  ViT-L/32 + ImageNet
# ===========================================================================
echo ""
echo "[5/6] Table 4 消融: ViT-L/32 ImageNet  $(date)"
IMAGE_SIZE=384 python experiments/scripts/run_ablation_study.py \
    --model vit-l-32 --dataset imagenet \
    --checkpoint /root/checkpoints/vit_l32_imagenet/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 50 \
    --hvp_batches 2 --batch_size 16 \
    --hvp_mode block --device cuda
echo "[5/6] 完成 $(date)"

# ===========================================================================
# [6/6]  Table 3 主结果  —  ViT-L/32 + ImageNet
# ===========================================================================
echo ""
echo "[6/6] Table 3 主结果: ViT-L/32 ImageNet  $(date)"
IMAGE_SIZE=384 python experiments/scripts/run_method_comparison.py \
    --model vit-l-32 --dataset imagenet \
    --checkpoint /root/checkpoints/vit_l32_imagenet/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 50 \
    --hvp_batches 2 --batch_size 16 \
    --hvp_mode block --device cuda
echo "[6/6] 完成 $(date)"

echo ""
echo "========================================="
echo " 全部 6 个新模型实验完成 - $(date)"
echo "========================================="
echo "结果保存在:"
echo "  Table 3 (主结果): results/paper_results/table1/"
echo "  Table 4 (消融):   results/paper_results/table3/"
