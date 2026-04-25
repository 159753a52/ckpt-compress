#!/bin/bash
# Block-wise HVP 近似精度实验
# 在远程服务器上运行: cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress && bash run_blockwise_hvp_experiments.sh
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="experiments/results/blockwise_hvp_${TIMESTAMP}.log"
mkdir -p experiments/results/blockwise_hvp_analysis

echo "============================================" | tee "$LOG"
echo "Block-wise HVP 实验 $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# 实验 A: GPT-2 Small, γ 扫描 (不带 normalize，论文默认配置)
echo "[Exp A] GPT-2 Small γ sweep (no normalize)..." | tee -a "$LOG"
python -m experiments.scripts.run_blockwise_hvp_analysis \
    --model gpt2-small \
    --dataset wikitext2 \
    --seq_lengths 128,256,512 \
    --hvp_batches 2 \
    --device cuda \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"

# 实验 B: GPT-2 Medium γ 扫描
echo "[Exp B] GPT-2 Medium γ sweep (no normalize)..." | tee -a "$LOG"
python -m experiments.scripts.run_blockwise_hvp_analysis \
    --model gpt2-medium \
    --dataset wikitext2 \
    --seq_lengths 128,256,512 \
    --hvp_batches 2 \
    --device cuda \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[ALL DONE] $(date)" | tee -a "$LOG"
