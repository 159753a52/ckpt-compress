#!/bin/bash
# 细粒度剪枝分析 - 快速启动脚本

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}细粒度剪枝分析 - 快速启动${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 检查检查点路径
CHECKPOINT_PATH="checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${YELLOW}警告: 未找到默认检查点${NC}"
    echo -e "${YELLOW}路径: $CHECKPOINT_PATH${NC}"
    echo ""
    echo "请指定检查点路径："
    read -p "检查点路径: " CHECKPOINT_PATH
fi

# 检查 CUDA 是否可用
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ 检测到 CUDA${NC}"
    DEVICE="cuda"
else
    echo -e "${YELLOW}⚠ 未检测到 CUDA，将使用 CPU（速度较慢）${NC}"
    DEVICE="cpu"
fi

echo ""
echo -e "${BLUE}实验配置:${NC}"
echo "  检查点: $CHECKPOINT_PATH"
echo "  设备: $DEVICE"
echo "  梯度累积步数: 100"
echo "  评估批次数: 10"
echo "  稀疏度: 5%-90% (18个点)"
echo ""
echo -e "${YELLOW}预计运行时间: 5.5-6.5 小时${NC}"
echo ""

# 询问是否继续
read -p "是否开始实验? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="results/fine_grained_pruning_analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo -e "${GREEN}开始实验...${NC}"
echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"
echo ""

# 运行实验
python experiments/scripts/fine_grained_pruning_analysis.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --num_steps 100 \
    --batch_size 4 \
    --seq_length 512 \
    --num_eval_batches 10 \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}实验完成！${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo "结果保存在: $OUTPUT_DIR"
    echo ""
    echo "查看结果:"
    echo "  1. 分析报告: cat $OUTPUT_DIR/ANALYSIS_REPORT.md"
    echo "  2. 热力图: open $OUTPUT_DIR/lossless_sparsity_heatmap.png"
    echo "  3. 对比图: open $OUTPUT_DIR/block_comparison.png"
    echo "  4. CSV数据: cat $OUTPUT_DIR/lossless_sparsity_summary.csv"
else
    echo ""
    echo -e "${YELLOW}实验失败，请检查错误信息${NC}"
    exit 1
fi
