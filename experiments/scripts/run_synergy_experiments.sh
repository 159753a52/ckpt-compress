#!/bin/bash
# ============================================================
# 综合实验：FT训练 + 剪枝-量化协同效应
# 
# Experiment 1 (30% sparsity): 基础FT + 协同效应对比
#   none, magnitude+uniform, magnitude+uniform+kmeans16,
#   ours-2d, ours-2d+kmeans16, ours-2d+kmeans256
#
# Experiment 2 (40% sparsity): 高压缩率对比
#   magnitude+uniform, magnitude+uniform+kmeans16,
#   ours-2d, ours-2d+kmeans16
#
# 预计总时长: ~100分钟 (A30 GPU)
# ============================================================
set -e

# --- 环境配置 ---
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate GPT-2

cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

CKPT="/lihongliang/fangzl/ckpt-compress/checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt"
OUTBASE="results/paper_results/fault_tolerant"

echo "========================================="
echo "实验开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================="

# ============================================================
# Experiment 1: 30% sparsity — FT + Synergy
# ============================================================
echo ""
echo ">>> [Exp 1/2] FT Synergy @ 30% sparsity (6 methods)"
echo "    开始: $(date '+%H:%M:%S')"

PYTHONUNBUFFERED=1 python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium --dataset wikitext103 \
    --checkpoint "$CKPT" \
    --total_steps 1000 --num_recoveries 10 \
    --prune_ratio 0.3 --seq_length 128 --batch_size 2 \
    --methods "none,magnitude+uniform,magnitude+uniform+kmeans16,ours-2d,ours-2d+kmeans16,ours-2d+kmeans256" \
    --eval_interval 10 --num_importance_steps 10 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "${OUTBASE}/synergy_30"

echo "    完成: $(date '+%H:%M:%S')"

# ============================================================
# Experiment 2: 40% sparsity — Higher compression
# ============================================================
echo ""
echo ">>> [Exp 2/2] FT Synergy @ 40% sparsity (4 methods)"
echo "    开始: $(date '+%H:%M:%S')"

PYTHONUNBUFFERED=1 python experiments/scripts/run_fault_tolerant_training.py \
    --model gpt2-medium --dataset wikitext103 \
    --checkpoint "$CKPT" \
    --total_steps 1000 --num_recoveries 10 \
    --prune_ratio 0.4 --seq_length 128 --batch_size 2 \
    --methods "magnitude+uniform,magnitude+uniform+kmeans16,ours-2d,ours-2d+kmeans16" \
    --eval_interval 10 --num_importance_steps 10 \
    --alpha 0.7 --protection_ratio 0.001 \
    --device cuda --output_dir "${OUTBASE}/synergy_40"

echo "    完成: $(date '+%H:%M:%S')"

# ============================================================
# Done
# ============================================================
echo ""
echo "========================================="
echo "全部实验完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "结果目录:"
echo "  30%: ${OUTBASE}/synergy_30/"
echo "  40%: ${OUTBASE}/synergy_40/"
echo "========================================="

# 打印最终摘要
echo ""
echo "--- 30% 结果摘要 ---"
CFG30=$(ls -1t ${OUTBASE}/synergy_30/*_config.json 2>/dev/null | head -n 1 || true)
if [[ -n "${CFG30}" ]]; then
  CFG_PATH="${CFG30}" python3 -c "
import os, json
path = os.environ['CFG_PATH']
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

final = data.get('final', {})
baseline = data.get('baseline', {})
bp = baseline.get('perplexity', None)
if isinstance(bp, (int, float)):
    print(f\"Baseline PPL: {bp:.2f}\")
else:
    print(f\"Baseline PPL: {bp}\")

for m, v in final.items():
    ppl = v.get('perplexity', v.get('loss', None))
    if isinstance(ppl, (int, float)):
        print(f\"  {m}: PPL={ppl:.2f}\")
    else:
        print(f\"  {m}: PPL={ppl}\")
"
else
  echo "(未找到配置文件)"
fi

echo ""
echo "--- 40% 结果摘要 ---"
CFG40=$(ls -1t ${OUTBASE}/synergy_40/*_config.json 2>/dev/null | head -n 1 || true)
if [[ -n "${CFG40}" ]]; then
  CFG_PATH="${CFG40}" python3 -c "
import os, json
path = os.environ['CFG_PATH']
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

final = data.get('final', {})
baseline = data.get('baseline', {})
bp = baseline.get('perplexity', None)
if isinstance(bp, (int, float)):
    print(f\"Baseline PPL: {bp:.2f}\")
else:
    print(f\"Baseline PPL: {bp}\")

for m, v in final.items():
    ppl = v.get('perplexity', v.get('loss', None))
    if isinstance(ppl, (int, float)):
        print(f\"  {m}: PPL={ppl:.2f}\")
    else:
        print(f\"  {m}: PPL={ppl}\")
"
else
  echo "(未找到配置文件)"
fi
