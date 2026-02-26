#!/bin/bash

# BERT-Large 模型下载和验证脚本
# 使用方法: bash verify_bert_installation.sh

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BERT-Large 模型安装验证"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. 检查下载脚本
echo "1️⃣  检查下载脚本..."
if [ -f "scripts/download_models.py" ]; then
    echo "   ✓ 下载脚本存在"
else
    echo "   ✗ 下载脚本不存在"
    exit 1
fi

# 2. 检查 BERT 模型文件
echo ""
echo "2️⃣  检查 BERT 模型文件..."
if [ -f "src/ckpt_compress/models/bert.py" ]; then
    echo "   ✓ BERT 模型文件存在"
else
    echo "   ✗ BERT 模型文件不存在"
    exit 1
fi

# 3. 检查文档
echo ""
echo "3️⃣  检查文档..."
docs=("docs/BERT_USAGE.md" "docs/BERT_INTEGRATION_SUMMARY.md" "BERT_QUICKSTART.md")
for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        echo "   ✓ $doc"
    else
        echo "   ✗ $doc 不存在"
    fi
done

# 4. 检查示例脚本
echo ""
echo "4️⃣  检查示例脚本..."
examples=("examples/test_bert_large.py" "examples/bert_quickstart.py")
for example in "${examples[@]}"; do
    if [ -f "$example" ]; then
        echo "   ✓ $example"
    else
        echo "   ✗ $example 不存在"
    fi
done

# 5. 检查模型缓存目录
echo ""
echo "5️⃣  检查模型缓存目录..."
if [ -d "data/models" ]; then
    echo "   ✓ 缓存目录存在: data/models"
    echo "   📁 目录内容:"
    ls -lh data/models/ 2>/dev/null | head -10
else
    echo "   ⚠️  缓存目录不存在，将在首次下载时创建"
fi

# 6. 测试 Python 导入
echo ""
echo "6️⃣  测试 Python 导入..."
python3 -c "
try:
    from src.ckpt_compress.models.bert import get_bert_large, get_bert_base
    print('   ✓ BERT 模块导入成功')
except ImportError as e:
    print(f'   ✗ 导入失败: {e}')
    exit(1)
" || exit 1

# 7. 检查依赖
echo ""
echo "7️⃣  检查依赖..."
python3 -c "
import sys
try:
    import transformers
    print(f'   ✓ transformers 版本: {transformers.__version__}')
except ImportError:
    print('   ✗ transformers 未安装')
    print('   💡 请运行: pip install transformers')
    sys.exit(1)

try:
    import torch
    print(f'   ✓ torch 版本: {torch.__version__}')
except ImportError:
    print('   ✗ torch 未安装')
    sys.exit(1)
" || exit 1

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ 验证完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 下一步操作:"
echo ""
echo "   1. 下载 BERT-Large 模型（如果尚未下载）:"
echo "      python scripts/download_models.py --model bert-large"
echo ""
echo "   2. 运行测试脚本:"
echo "      python examples/test_bert_large.py"
echo ""
echo "   3. 运行快速开始示例:"
echo "      python examples/bert_quickstart.py --all"
echo ""
echo "   4. 查看详细文档:"
echo "      cat docs/BERT_USAGE.md"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
