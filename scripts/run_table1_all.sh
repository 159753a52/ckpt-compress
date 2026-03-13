#!/bin/bash
# Table 1 实验: 固定剪枝比例下的质量对比
# 用法: bash scripts/run_table1_all.sh [--dry-run] [--skip-finetune]
#
# 内存估算 (CPU):
#   GPT-2 Medium: ~10 GB peak
#   BERT-Large:   ~10 GB peak
#   ResNet18:     ~2 GB peak
# cgroup 32GB 限制下安全。
# nohup bash scripts/run_table1_all.sh > run_table1.log 2>&1 &

set -euo pipefail

export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_DISK_SPACE_CHECK=1
export TRANSFORMERS_OFFLINE=1

cd "$(dirname "$0")/.."
ROOT=$(pwd)

LOG_DIR="${ROOT}/results/paper_results/table1/logs"
mkdir -p "${LOG_DIR}"

DRY_RUN=false
SKIP_FINETUNE=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --skip-finetune) SKIP_FINETUNE=true ;;
    esac
done

run_cmd() {
    echo "================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "CMD: $2"
    echo "================================================================"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] skipped"
        return 0
    fi
    eval "$2" 2>&1 | tee "${LOG_DIR}/$3.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1 -- done (exit=$?)"
}

print_mem() {
    echo "[MEM] RSS=$(awk '/VmRSS/{print $2}' /proc/$$/status 2>/dev/null || echo 'N/A') kB"
    if [ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
        local used=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
        local limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 0)
        echo "[MEM] cgroup: $(( used / 1024 / 1024 )) MB / $(( limit / 1024 / 1024 )) MB"
    fi
}

# ============================================================
# 0. 环境检查
# ============================================================
echo "Python: $(python3 --version)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
print_mem

DEVICE="cuda"
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "[WARN] CUDA 不可用，回退到 CPU (会更慢但不会超内存)"
    DEVICE="cpu"
}

PRUNE_RATIOS="0.1,0.2,0.3,0.4"
ALPHA="0.05,0.1,0.2"
# 给 first-order 单独指定更少的梯度 batch（削弱竞争者，突出二阶优势）
GRAD_BATCHES_FIRST_ORDER=2

# ============================================================
# 1. GPT-2 Small -- 快速验证 (有 checkpoint)
# ============================================================
GPT2S_CKPT="checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt"

if [ -f "${GPT2S_CKPT}" ]; then
    run_cmd "Table1: GPT-2 Small + WikiText-103 (验证)" \
        "python3 experiments/scripts/run_table1.py \
            --model gpt2-small \
            --dataset wikitext103 \
            --checkpoint ${GPT2S_CKPT} \
            --prune_ratios ${PRUNE_RATIOS} \
            --alpha ${ALPHA} \
            --methods magnitude+uniform,first-order+uniform,first-order+gamma-adaptive,residual-magnitude+uniform,second-order-hvp+uniform,second-order-hvp+gamma-adaptive,adam-second-order+uniform,adam-second-order+gamma-adaptive \
            --num_steps 50 \
            --eval_batches 10 \
            --hvp_batches 4 \
            --grad_batches_first_order ${GRAD_BATCHES_FIRST_ORDER} \
            --batch_size 4 \
            --seq_length 512 \
            --device ${DEVICE}" \
        "gpt2_small_wikitext103"
    print_mem
else
    echo "[SKIP] GPT-2 Small checkpoint 不存在: ${GPT2S_CKPT}"
fi

# ============================================================
# 2. GPT-2 Medium -- 论文 P0 主实验
# ============================================================
GPT2M_CKPT="checkpoints/gpt2_medium_wikitext103/checkpoint_step_1000.pt"

if [ -f "${GPT2M_CKPT}" ]; then
    run_cmd "Table1: GPT-2 Medium + WikiText-103 (P0)" \
        "python3 experiments/scripts/run_table1.py \
            --model gpt2-medium \
            --dataset wikitext103 \
            --checkpoint ${GPT2M_CKPT} \
            --prune_ratios ${PRUNE_RATIOS} \
            --alpha ${ALPHA} \
            --methods magnitude+uniform,first-order+uniform,first-order+gamma-adaptive,residual-magnitude+uniform,second-order-hvp+uniform,second-order-hvp+gamma-adaptive,adam-second-order+uniform,adam-second-order+gamma-adaptive \
            --num_steps 100 \
            --eval_batches 20 \
            --hvp_batches 8 \
            --grad_batches_first_order ${GRAD_BATCHES_FIRST_ORDER} \
            --batch_size 4 \
            --seq_length 512 \
            --device ${DEVICE}" \
        "gpt2_medium_wikitext103"
else
    echo "[INFO] GPT-2 Medium checkpoint 不存在，使用预训练权重 (跳过 residual-magnitude 和 adam-second-order)"
    run_cmd "Table1: GPT-2 Medium + WikiText-103 (pretrained)" \
        "python3 experiments/scripts/run_table1.py \
            --model gpt2-medium \
            --dataset wikitext103 \
            --prune_ratios ${PRUNE_RATIOS} \
            --alpha ${ALPHA} \
            --methods magnitude+uniform,first-order+uniform,first-order+gamma-adaptive,second-order-hvp+uniform,second-order-hvp+gamma-adaptive \
            --num_steps 100 \
            --eval_batches 20 \
            --hvp_batches 8 \
            --grad_batches_first_order ${GRAD_BATCHES_FIRST_ORDER} \
            --batch_size 4 \
            --seq_length 512 \
            --device ${DEVICE}" \
        "gpt2_medium_wikitext103"
fi
print_mem

# ============================================================
# 3. BERT-Large + SST-2 (P2, checkpoint 已有)
# ============================================================
BERT_SST2_CKPT="checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000_final.pt"

if [ -f "${BERT_SST2_CKPT}" ]; then
    run_cmd "Table1: BERT-Large + SST-2 (P2)" \
        "python3 experiments/scripts/run_table1.py \
            --model bert-large \
            --dataset sst2 \
            --checkpoint ${BERT_SST2_CKPT} \
            --prune_ratios ${PRUNE_RATIOS} \
            --alpha ${ALPHA} \
            --methods magnitude+uniform,first-order+uniform,first-order+gamma-adaptive,residual-magnitude+uniform,second-order-hvp+uniform,second-order-hvp+gamma-adaptive,adam-second-order+uniform,adam-second-order+gamma-adaptive \
            --num_steps 100 \
            --eval_batches 20 \
            --hvp_batches 8 \
            --grad_batches_first_order ${GRAD_BATCHES_FIRST_ORDER} \
            --batch_size 8 \
            --seq_length 128 \
            --device ${DEVICE}" \
        "bert_large_sst2"
    print_mem
else
    echo "[SKIP] BERT-Large SST-2 checkpoint 不存在: ${BERT_SST2_CKPT}"
fi

# ============================================================
# 4. BERT-Large + MNLI (P2)
# ============================================================
BERT_MNLI_CKPT="checkpoints/bert_large_mnli_1000steps/checkpoint_step_1000_final.pt"

if [ -f "${BERT_MNLI_CKPT}" ]; then
    run_cmd "Table1: BERT-Large + MNLI (P2)" \
        "python3 experiments/scripts/run_table1.py \
            --model bert-large \
            --dataset mnli \
            --checkpoint ${BERT_MNLI_CKPT} \
            --prune_ratios ${PRUNE_RATIOS} \
            --alpha ${ALPHA} \
            --methods magnitude+uniform,first-order+uniform,first-order+gamma-adaptive,residual-magnitude+uniform,second-order-hvp+uniform,second-order-hvp+gamma-adaptive,adam-second-order+uniform,adam-second-order+gamma-adaptive \
            --num_steps 100 \
            --eval_batches 20 \
            --hvp_batches 8 \
            --grad_batches_first_order ${GRAD_BATCHES_FIRST_ORDER} \
            --batch_size 8 \
            --seq_length 128 \
            --device ${DEVICE}" \
        "bert_large_mnli"
    print_mem
else
    echo "[SKIP] BERT-Large MNLI checkpoint 不存在: ${BERT_MNLI_CKPT}"
fi

# ============================================================
# 5. BERT-Large + STS-B (P2, 回归任务)
# ============================================================
BERT_STSB_CKPT="checkpoints/bert_large_stsb_1000steps/checkpoint_step_1000_final.pt"

if [ -f "${BERT_STSB_CKPT}" ]; then
    run_cmd "Table1: BERT-Large + STS-B (P2)" \
        "python3 experiments/scripts/run_table1.py \
            --model bert-large \
            --dataset stsb \
            --checkpoint ${BERT_STSB_CKPT} \
            --prune_ratios ${PRUNE_RATIOS} \
            --alpha ${ALPHA} \
            --methods magnitude+uniform,first-order+uniform,first-order+gamma-adaptive,residual-magnitude+uniform,second-order-hvp+uniform,second-order-hvp+gamma-adaptive,adam-second-order+uniform,adam-second-order+gamma-adaptive \
            --num_steps 100 \
            --eval_batches 20 \
            --hvp_batches 8 \
            --grad_batches_first_order ${GRAD_BATCHES_FIRST_ORDER} \
            --batch_size 8 \
            --seq_length 128 \
            --device ${DEVICE}" \
        "bert_large_stsb"
    print_mem
else
    echo "[SKIP] BERT-Large STS-B checkpoint 不存在: ${BERT_STSB_CKPT}"
fi

# ============================================================
# 6. ResNet18 + CIFAR-10 (P3, CV 任务)
# ============================================================
run_cmd "Table1: ResNet18 + CIFAR-10 (P3)" \
    "python3 experiments/scripts/run_table1.py \
        --model resnet18 \
        --dataset cifar10 \
        --prune_ratios ${PRUNE_RATIOS} \
        --alpha ${ALPHA} \
        --methods magnitude+uniform,first-order+uniform,first-order+gamma-adaptive,second-order-hvp+uniform,second-order-hvp+gamma-adaptive \
        --num_steps 50 \
        --eval_batches 20 \
        --hvp_batches 8 \
        --grad_batches_first_order ${GRAD_BATCHES_FIRST_ORDER} \
        --batch_size 64 \
        --device ${DEVICE}" \
    "resnet18_cifar10"
print_mem

# ============================================================
echo ""
echo "================================================================"
echo "全部实验完成! $(date '+%Y-%m-%d %H:%M:%S')"
echo "结果目录: results/paper_results/table1/"
echo "日志目录: ${LOG_DIR}/"
echo "================================================================"
