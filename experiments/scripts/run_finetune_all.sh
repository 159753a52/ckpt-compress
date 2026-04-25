#!/bin/bash
# ============================================================
# 三个模型一键微调脚本
# 依赖：先 source experiments/scripts/setup_env.sh
# ============================================================

set -e

LOG_DIR="/root/finetune_logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  论文实验：三模型微调"
echo "  开始时间: $(date)"
echo "============================================"

# ---------- 1. GPT-2 Medium / WikiText-103 ----------
echo ""
echo "[1/3] GPT-2 Medium / WikiText-103 微调..."
echo "  输出: /root/checkpoints/gpt2_medium_wikitext103"
python experiments/scripts/finetune/finetune_gpt2_medium.py \
    --total_steps 1000 \
    --batch_size 4 \
    --seq_length 512 \
    --lr 2e-5 \
    --save_every 200 \
    --device cuda \
    --output_dir /root/checkpoints/gpt2_medium_wikitext103 \
    --use_amp \
    2>&1 | tee "$LOG_DIR/gpt2m_wikitext103.log"

echo "[1/3] GPT-2 Medium 微调完成: $(date)"

# ---------- 2. Pythia-410M / Alpaca ----------
echo ""
echo "[2/3] Pythia-410M / Alpaca 微调..."
echo "  输出: /root/checkpoints/pythia_410m_alpaca"
python experiments/scripts/finetune/finetune_pythia_410m.py \
    --total_steps 1000 \
    --batch_size 4 \
    --seq_length 512 \
    --lr 2e-5 \
    --save_every 200 \
    --device cuda \
    --output_dir /root/checkpoints/pythia_410m_alpaca \
    --use_amp \
    2>&1 | tee "$LOG_DIR/pythia_alpaca.log"

echo "[2/3] Pythia-410M 微调完成: $(date)"

# ---------- 3. ViT-L/32 / ImageNet ----------
echo ""
echo "[3/3] ViT-L/32 / ImageNet 微调..."
echo "  输出: /root/checkpoints/vit_l32_imagenet"
echo "  ImageNet 数据: $IMAGENET_DATA_DIR"
# ViT-L/32-384 使用 384 分辨率
export IMAGE_SIZE=384
python experiments/scripts/finetune/finetune_vit.py \
    --model vit-l-32 \
    --total_steps 1000 \
    --batch_size 16 \
    --lr 2e-5 \
    --save_every 200 \
    --device cuda \
    --output_dir /root/checkpoints/vit_l32_imagenet \
    --data_dir "$IMAGENET_DATA_DIR" \
    --num_workers 4 \
    --use_amp \
    2>&1 | tee "$LOG_DIR/vit_l32_imagenet.log"

echo "[3/3] ViT-L/32 微调完成: $(date)"

# ---------- 汇总 ----------
echo ""
echo "============================================"
echo "  全部微调完成"
echo "  结束时间: $(date)"
echo "============================================"
echo ""
echo "Checkpoints:"
ls -lh /root/checkpoints/gpt2_medium_wikitext103/ 2>/dev/null
ls -lh /root/checkpoints/pythia_410m_alpaca/ 2>/dev/null
ls -lh /root/checkpoints/vit_l32_imagenet/ 2>/dev/null
