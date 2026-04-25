#!/bin/bash
# =============================================================================
# 论文 Table 3 (主结果) + Table 4 (消融) 全部实验
# 用法: nohup bash experiments/scripts/run_all_paper_experiments.sh > paper_exp.log 2>&1 &
# =============================================================================
set -e

# ---------- 环境 ----------
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate GPT-2
unset http_proxy https_proxy
export HF_ENDPOINT=https://hf-mirror.com

cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

echo "========================================="
echo " 论文实验批次启动 - $(date)"
echo "========================================="

# ===========================================================================
# [1/6]  Table 4 消融  —  BERT-Large + SST-2
# ===========================================================================
echo ""
echo "[1/6] Table 4 消融: BERT-Large SST-2  $(date)"
python experiments/scripts/run_ablation_study.py \
    --model bert-large --dataset sst2 \
    --checkpoint /root/checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 100 \
    --hvp_batches 2 --seq_length 128 --batch_size 8 \
    --hvp_mode block --device cuda
echo "[1/6] 完成 $(date)"

# ===========================================================================
# [2/6]  Table 3 主结果  —  BERT-Large + SST-2
# ===========================================================================
echo ""
echo "[2/6] Table 3 主结果: BERT-Large SST-2  $(date)"
python experiments/scripts/run_method_comparison.py \
    --model bert-large --dataset sst2 \
    --checkpoint /root/checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 100 \
    --hvp_batches 2 --seq_length 128 --batch_size 8 \
    --hvp_mode block --device cuda
echo "[2/6] 完成 $(date)"

# ===========================================================================
# [3/6]  Table 4 消融  —  BERT-Large + MNLI
# ===========================================================================
echo ""
echo "[3/6] Table 4 消融: BERT-Large MNLI  $(date)"
python experiments/scripts/run_ablation_study.py \
    --model bert-large --dataset mnli \
    --checkpoint /root/checkpoints/bert_large_mnli_1000steps/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 100 \
    --hvp_batches 2 --seq_length 128 --batch_size 8 \
    --hvp_mode block --device cuda
echo "[3/6] 完成 $(date)"

# ===========================================================================
# [4/6]  Table 3 主结果  —  BERT-Large + MNLI
# ===========================================================================
echo ""
echo "[4/6] Table 3 主结果: BERT-Large MNLI  $(date)"
python experiments/scripts/run_method_comparison.py \
    --model bert-large --dataset mnli \
    --checkpoint /root/checkpoints/bert_large_mnli_1000steps/checkpoint_step_1000.pt \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 100 \
    --hvp_batches 2 --seq_length 128 --batch_size 8 \
    --hvp_mode block --device cuda
echo "[4/6] 完成 $(date)"

# ===========================================================================
# [5/6]  Table 4 消融  —  GPT-2 Medium + WikiText-2
# ===========================================================================
echo ""
echo "[5/6] Table 4 消融: GPT-2 Medium WikiText-2  $(date)"
python experiments/scripts/run_ablation_study.py \
    --model gpt2-medium --dataset wikitext2 \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 50 \
    --hvp_batches 2 --seq_length 128 --batch_size 2 \
    --hvp_mode block --device cuda
echo "[5/6] 完成 $(date)"

# ===========================================================================
# [6/6]  Table 3 主结果  —  GPT-2 Medium + WikiText-2
#   注: 无 checkpoint，residual-magnitude 自动跳过
# ===========================================================================
echo ""
echo "[6/6] Table 3 主结果: GPT-2 Medium WikiText-2  $(date)"
python experiments/scripts/run_method_comparison.py \
    --model gpt2-medium --dataset wikitext2 \
    --prune_ratios 0.2,0.3,0.4 \
    --alpha 0.5 --num_steps 50 --eval_batches 50 \
    --hvp_batches 2 --seq_length 128 --batch_size 2 \
    --hvp_mode block --device cuda
echo "[6/6] 完成 $(date)"

echo ""
echo "========================================="
echo " 全部 6 个实验完成 - $(date)"
echo "========================================="
echo "结果保存在:"
echo "  Table 3: results/paper_results/table1/"
echo "  Table 4: results/paper_results/table3/"
