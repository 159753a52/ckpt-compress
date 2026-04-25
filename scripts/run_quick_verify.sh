#!/bin/bash
# 快速验证 optimizer fix — K=2 小测试
set -e
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

echo "========================================="
echo "Quick Fix Verification (K=2, 200 steps)"
echo "========================================="

# Test with GPT-2 Medium — should see PPL stay near baseline (~18.07)
python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium --dataset wikitext103 \
    --checkpoint checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt \
    --total_steps 200 --num_recoveries 2 \
    --prune_ratio 0.3 \
    --seq_length 512 --batch_size 4 \
    --lr 5e-5 --lr_schedule cosine \
    --methods "none,magnitude+uniform,ours-2d" \
    --eval_interval 20 --num_importance_steps 5 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda \
    --output_dir "results/paper_results/fix_verify"

echo ""
echo "========================================="
echo "Quick test done!"
echo "========================================="
