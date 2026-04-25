#!/bin/bash
# 主实验队列：按优先级依次执行
# 使用方法：
#   nohup bash scripts/run_priority_queue.sh > logs/priority_queue.log 2>&1 &
#
# 优先级：
# 1. ours-2d 验证（BERT SST-2 + GPT-2）
# 2. GPT-2 K=10 多稀疏度 Sweep（Table 1 验证）
# 3. GPT-2 K=10 Ablation（Table 2 验证）

set -e
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

mkdir -p logs

echo "============================================================"
echo "Priority Queue Started: $(date)"
echo "============================================================"

# --- Priority 1: ours-2d 验证 ---
echo ""
echo "############################################################"
echo "PRIORITY 1: ours-2d Validation"
echo "############################################################"
bash scripts/run_test_ours2d.sh

echo ""
echo "############################################################"
echo "PRIORITY 2: GPT-2 K=10 Sweep (Table 1)"
echo "############################################################"
bash scripts/run_gpt2_k10_sweep.sh

echo ""
echo "############################################################"
echo "PRIORITY 3: GPT-2 K=10 Ablation (Table 2)"
echo "############################################################"
bash scripts/run_gpt2_k10_ablation.sh

echo ""
echo "============================================================"
echo "All Priority Queue Experiments Done: $(date)"
echo "============================================================"
