#!/bin/bash
# 测试减去 0.4 的剪枝配置

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}剪枝配置测试 - 减去 0.4${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

CHECKPOINT_PATH="checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${YELLOW}警告: 未找到默认检查点${NC}"
    echo "请指定检查点路径："
    read -p "检查点路径: " CHECKPOINT_PATH
fi

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ 检测到 CUDA${NC}"
    DEVICE="cuda"
else
    echo -e "${YELLOW}⚠ 未检测到 CUDA，将使用 CPU${NC}"
    DEVICE="cpu"
fi

echo ""
echo -e "${BLUE}剪枝配置:${NC}"
echo "  基于无损剪枝比例，每项减去 0.4（低于 0 则设为 0）"
echo ""
echo "  示例 (Block 0):"
echo "    - attn_qkv:  35% →  0% (35% - 40% < 0)"
echo "    - attn_proj: 85% → 45% (85% - 40%)"
echo "    - mlp_fc:    55% → 15% (55% - 40%)"
echo "    - mlp_proj:  55% → 15% (55% - 40%)"
echo "    - ln_1:      25% →  0% (25% - 40% < 0)"
echo "    - ln_2:      20% →  0% (20% - 40% < 0)"
echo ""
echo -e "${BLUE}统计信息:${NC}"
echo "  原始平均无损剪枝率: 44.4%"
echo "  调整后平均剪枝率: 12.5%"
echo "  变为 0 的层数: 36/72 (50%)"
echo ""
echo -e "${YELLOW}预计运行时间: 10-15 分钟${NC}"
echo ""

read -p "是否开始测试? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

OUTPUT_DIR="results/pruning_test_minus0.4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo -e "${GREEN}开始测试...${NC}"
echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"
echo ""

python experiments/scripts/test_pruning_config.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config experiments/configs/pruning_config_minus0.4.json \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}测试完成！${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo "结果保存在: $OUTPUT_DIR"

    if [ -f "$OUTPUT_DIR/summary.json" ]; then
        echo ""
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
        print(f"  结论: ✅ 无损剪枝")
    elif data['loss_increase'] <= 0.01:
        print(f"  结论: ⚠️ 轻微损失")
    else:
        print(f"  结论: ❌ 明显损失")
EOF
    fi
fi
