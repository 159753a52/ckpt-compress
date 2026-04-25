#!/bin/bash
# ================================================================
# 夜间批量实验：Fault-Tolerant Training K=10
# 预计总耗时：8-10 小时
# 运行方式：nohup bash scripts/overnight_ft_batch.sh > logs/overnight_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ================================================================

# 不用 set -e，避免单个实验失败中断全部
FAIL_COUNT=0

# 环境激活
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_DATASETS_CACHE=/lihongliang/fangzl/.cache/huggingface/datasets
export ALPACA_LOCAL_PATH="/lihongliang/fangzl/ckpt-compress/data/alpaca/alpaca_data.json"
export IMAGENET_DATA_DIR="/lihongliang/bobzhou/dataset"
export IMAGE_SIZE=384

cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress
mkdir -p logs

OUTDIR="results/paper_results/fault_tolerant"
METHODS="none,magnitude+uniform,ours-2d"

echo "=========================================="
echo "夜间批量实验开始: $(date)"
echo "=========================================="

# ----------------------------------------------------------------
# 1. Pythia-410M / Alpaca K=10 (预计 ~3 小时)
#    1000 步，3 方法 × 4 ratio
# ----------------------------------------------------------------
echo ""
echo ">>> [1/12] Pythia-410M Alpaca K=10 ratio=0.1 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model pythia-410m --dataset alpaca \
    --checkpoint checkpoints/pythia_410m_alpaca/checkpoint_step_1000.pt \
    --total_steps 1000 --num_recoveries 10 \
    --prune_ratio 0.1 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 20 \
    --batch_size 2 --seq_length 128 --lr 2e-5 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --lr_schedule cosine --seed 42

echo ">>> [2/12] Pythia-410M Alpaca K=10 ratio=0.2 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model pythia-410m --dataset alpaca \
    --checkpoint checkpoints/pythia_410m_alpaca/checkpoint_step_1000.pt \
    --total_steps 1000 --num_recoveries 10 \
    --prune_ratio 0.2 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 20 \
    --batch_size 2 --seq_length 128 --lr 2e-5 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --lr_schedule cosine --seed 42

echo ">>> [3/12] Pythia-410M Alpaca K=10 ratio=0.3 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model pythia-410m --dataset alpaca \
    --checkpoint checkpoints/pythia_410m_alpaca/checkpoint_step_1000.pt \
    --total_steps 1000 --num_recoveries 10 \
    --prune_ratio 0.3 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 20 \
    --batch_size 2 --seq_length 128 --lr 2e-5 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --lr_schedule cosine --seed 42

echo ">>> [4/12] Pythia-410M Alpaca K=10 ratio=0.4 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model pythia-410m --dataset alpaca \
    --checkpoint checkpoints/pythia_410m_alpaca/checkpoint_step_1000.pt \
    --total_steps 1000 --num_recoveries 10 \
    --prune_ratio 0.4 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 20 \
    --batch_size 2 --seq_length 128 --lr 2e-5 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --lr_schedule cosine --seed 42

# ----------------------------------------------------------------
# 2. ViT-L/32 / ImageNet K=10 (预计 ~2.5 小时)
#    1000 步，3 方法 × 4 ratio
# ----------------------------------------------------------------
echo ""
echo ">>> [5/12] ViT-L/32 ImageNet K=10 ratio=0.1 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model vit-l-32 --dataset imagenet \
    --checkpoint checkpoints/vit_large_imagenet/checkpoint_step_1000.pt \
    --total_steps 1000 --num_recoveries 10 \
    --prune_ratio 0.1 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 20 \
    --batch_size 4 --lr 2e-5 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --lr_schedule cosine --seed 42

echo ">>> [6/12] ViT-L/32 ImageNet K=10 ratio=0.2 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model vit-l-32 --dataset imagenet \
    --checkpoint checkpoints/vit_large_imagenet/checkpoint_step_1000.pt \
    --total_steps 1000 --num_recoveries 10 \
    --prune_ratio 0.2 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 20 \
    --batch_size 4 --lr 2e-5 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --lr_schedule cosine --seed 42

echo ">>> [7/12] ViT-L/32 ImageNet K=10 ratio=0.3 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model vit-l-32 --dataset imagenet \
    --checkpoint checkpoints/vit_large_imagenet/checkpoint_step_1000.pt \
    --total_steps 1000 --num_recoveries 10 \
    --prune_ratio 0.3 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 20 \
    --batch_size 4 --lr 2e-5 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --lr_schedule cosine --seed 42

echo ">>> [8/12] ViT-L/32 ImageNet K=10 ratio=0.4 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model vit-l-32 --dataset imagenet \
    --checkpoint checkpoints/vit_large_imagenet/checkpoint_step_1000.pt \
    --total_steps 1000 --num_recoveries 10 \
    --prune_ratio 0.4 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 20 \
    --batch_size 4 --lr 2e-5 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --lr_schedule cosine --seed 42

# ----------------------------------------------------------------
# 3. GPT-2 Medium / WikiText-103 K=10 (预计 ~4 小时)
#    2000 步，3 方法 × 4 ratio
#    random_init 与已有实验保持一致
# ----------------------------------------------------------------
echo ""
echo ">>> [9/12] GPT-2 Medium WikiText-103 K=10 ratio=0.1 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium --dataset wikitext103 \
    --total_steps 2000 --num_recoveries 10 \
    --prune_ratio 0.1 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 50 \
    --batch_size 2 --seq_length 128 --lr 3e-4 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --random_init --lr_schedule cosine --seed 42

echo ">>> [10/12] GPT-2 Medium WikiText-103 K=10 ratio=0.2 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium --dataset wikitext103 \
    --total_steps 2000 --num_recoveries 10 \
    --prune_ratio 0.2 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 50 \
    --batch_size 2 --seq_length 128 --lr 3e-4 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --random_init --lr_schedule cosine --seed 42

echo ">>> [11/12] GPT-2 Medium WikiText-103 K=10 ratio=0.3 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium --dataset wikitext103 \
    --total_steps 2000 --num_recoveries 10 \
    --prune_ratio 0.3 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 50 \
    --batch_size 2 --seq_length 128 --lr 3e-4 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --random_init --lr_schedule cosine --seed 42

echo ">>> [12/12] GPT-2 Medium WikiText-103 K=10 ratio=0.4 @ $(date)"
python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium --dataset wikitext103 \
    --total_steps 2000 --num_recoveries 10 \
    --prune_ratio 0.4 --methods "$METHODS" \
    --num_importance_steps 10 --eval_interval 50 \
    --batch_size 2 --seq_length 128 --lr 3e-4 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "$OUTDIR" \
    --random_init --lr_schedule cosine --seed 42

echo ""
echo "=========================================="
echo "全部实验完成: $(date)"
echo "=========================================="
echo "结果保存在: $OUTDIR"
ls -lt "$OUTDIR"/*.json | head -20
