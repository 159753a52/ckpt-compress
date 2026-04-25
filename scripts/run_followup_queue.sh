#!/bin/bash
# ============================================================
# 跟进实验队列：在 priority_queue 完成后运行
#
# 内容:
# 1. Table 3 Synergy (pruning+quantization)
# 2. BERT-Large SST-2 K=10 Sweep (补充数据)
# 3. BERT MNLI K=5 p=0.4 补跑 (之前数据丢失)
#
# 使用:
#   nohup bash scripts/run_followup_queue.sh > logs/followup_queue.log 2>&1 &
# ============================================================
set -e
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate fangzl
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

mkdir -p logs

echo "============================================================"
echo "Followup Queue Started: $(date)"
echo "============================================================"

# --- 1. Table 3 Synergy ---
echo ""
echo "############################################################"
echo "TASK 1: Table 3 Synergy (Prune+Quant)"
echo "############################################################"
bash scripts/run_gpt2_k10_synergy.sh

# --- 2. BERT-Large SST-2 K=10 ---
echo ""
echo "############################################################"
echo "TASK 2: BERT-Large SST-2 K=10 Sweep"
echo "############################################################"
bash scripts/run_bert_k10_sweep.sh

# --- 3. BERT MNLI p=0.4 补跑 ---
echo ""
echo "############################################################"
echo "TASK 3: BERT MNLI K=5 p=0.4 补跑"
echo "############################################################"
python experiments/scripts/run_fault_tolerant_training.py \
    --model bert-large --dataset mnli \
    --checkpoint "checkpoints/bert_large_mnli_1000steps/checkpoint_step_1000_final.pt" \
    --total_steps 1000 --num_recoveries 5 \
    --prune_ratio 0.4 \
    --seq_length 128 --batch_size 4 \
    --lr 1e-5 --lr_schedule cosine \
    --methods "none,magnitude+uniform,first-order+uniform,magnitude+weibull-adaptive" \
    --eval_interval 20 --num_importance_steps 10 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda \
    --output_dir "results/paper_results/bert_mnli_k5_p04_rerun"

echo ""
echo "============================================================"
echo "All Followup Experiments Done: $(date)"
echo "============================================================"
