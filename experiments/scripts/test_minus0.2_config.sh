#!/bin/bash
# 测试基于分析结果每项减去 0.2 的剪枝配置

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}剪枝配置测试 - 所有 Blocks（每项减 0.2）${NC}"
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
echo -e "${BLUE}剪枝配置:${NC}"
echo "  基于 lossless_sparsity_summary.csv 的无损剪枝比例"
echo "  每项减去 0.2（如果低于 0 则设为 0）"
echo ""
echo "  示例 (Block 0):"
echo "    - attn_qkv:  35% → 15% (35% - 20%)"
echo "    - attn_proj: 85% → 65% (85% - 20%)"
echo "    - mlp_fc:    55% → 35% (55% - 20%)"
echo "    - mlp_proj:  55% → 35% (55% - 20%)"
echo "    - ln_1:      25% →  5% (25% - 20%)"
echo "    - ln_2:      20% →  0% (20% - 20% = 0%)"
echo ""
echo -e "${BLUE}实验参数:${NC}"
echo "  检查点: $CHECKPOINT_PATH"
echo "  设备: $DEVICE"
echo "  梯度累积步数: 100"
echo "  评估批次数: 10"
echo "  配置文件: experiments/configs/pruning_config_minus0.2.json"
echo ""
echo -e "${BLUE}预期结果:${NC}"
echo "  原始平均无损剪枝率: 44.4%"
echo "  调整后平均剪枝率: 26.0%"
echo ""
echo -e "${YELLOW}预计运行时间: 10-15 分钟${NC}"
echo ""

# 询问是否继续
read -p "是否开始测试? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="results/pruning_test_minus0.2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo -e "${GREEN}开始测试...${NC}"
echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"
echo ""

# 运行测试
python experiments/scripts/test_pruning_config.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config experiments/configs/pruning_config_minus0.2.json \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}测试完成！${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo "结果保存在: $OUTPUT_DIR"
    echo ""
    echo "查看结果:"
    echo "  1. 测试报告: cat $OUTPUT_DIR/REPORT.md"
    echo "  2. 结果摘要: cat $OUTPUT_DIR/summary.json"
    echo "  3. 详细统计: cat $OUTPUT_DIR/pruning_stats.csv"
    echo ""

    # 显示关键结果
    if [ -f "$OUTPUT_DIR/summary.json" ]; then
        echo -e "${BLUE}关键结果:${NC}"
        python3 << EOF
import json
with open('$OUTPUT_DIR/summary.json', 'r') as f:
    data = json.load(f)
    print(f"  基线损失: {data['baseline_loss']:.6f}")
    print(f"  剪枝后损失: {data['pruned_loss']:.6f}")
    print(f"  损失增量: {data['loss_increase']:+.6f}")
    print(f"  整体稀疏度: {data['overall_sparsity']*100:.2f}%")
    if data['loss_increase'] <= 0.001:
        print(f"  结论: ✅ 无损剪枝（损失增量 ≤ 0.001）")
    elif data['loss_increase'] <= 0.01:
        print(f"  结论: ⚠️ 轻微损失（损失增量 ≤ 0.01）")
    else:
        print(f"  结论: ❌ 明显损失（损失增量 > 0.01）")
EOF
    fi
else
    echo ""
    echo -e "${YELLOW}测试失败，请检查错误信息${NC}"
    exit 1
fi
