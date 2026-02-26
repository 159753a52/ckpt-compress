"""
测试 BERT-Large 模型的加载和基本使用。

这个脚本演示如何：
1. 下载 BERT-Large 预训练模型
2. 加载模型并查看配置
3. 进行简单的前向传播测试
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ckpt_compress.models.bert import get_bert_large


def test_bert_large_download_and_load():
    """测试 BERT-Large 模型的下载和加载。"""
    print("="*60)
    print("测试 BERT-Large 模型")
    print("="*60)

    # 1. 加载预训练模型（会自动下载）
    print("\n1. 加载 BERT-Large 预训练模型...")
    print("   注意：首次运行会从 HuggingFace 下载约 1.3GB 的模型文件")

    try:
        model = get_bert_large(
            pretrained=True,
            cache_dir="./data/models"
        )
        print("   ✓ 模型加载成功！")
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
        return False

    # 2. 查看模型配置
    print("\n2. 模型配置信息:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"   - {key}: {value}")

    # 3. 计算参数量
    print("\n3. 模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - 总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   - 可训练参数: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    # 4. 测试前向传播
    print("\n4. 测试前向传播:")
    model.eval()

    # 创建虚拟输入
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    print(f"   - 输入形状: {input_ids.shape}")

    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            print(f"   - 输出形状: {logits.shape}")
            print(f"   - 输出范围: [{logits.min():.2f}, {logits.max():.2f}]")
            print("   ✓ 前向传播成功！")
        except Exception as e:
            print(f"   ✗ 前向传播失败: {e}")
            return False

    # 5. 测试状态字典
    print("\n5. 测试状态字典:")
    state_dict = model.state_dict()
    print(f"   - 状态字典包含 {len(state_dict)} 个张量")
    print(f"   - 前 5 个键: {list(state_dict.keys())[:5]}")

    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)

    return True


def test_bert_base_comparison():
    """对比 BERT-Base 和 BERT-Large。"""
    print("\n" + "="*60)
    print("对比 BERT-Base 和 BERT-Large")
    print("="*60)

    from src.ckpt_compress.models.bert import get_bert_base

    # 加载两个模型（不使用预训练权重以节省时间）
    print("\n加载模型（随机初始化）...")
    bert_base = get_bert_base(pretrained=False)
    bert_large = get_bert_large(pretrained=False)

    # 对比参数量
    base_params = sum(p.numel() for p in bert_base.parameters())
    large_params = sum(p.numel() for p in bert_large.parameters())

    print(f"\n参数量对比:")
    print(f"  BERT-Base:  {base_params:,} ({base_params/1e6:.1f}M)")
    print(f"  BERT-Large: {large_params:,} ({large_params/1e6:.1f}M)")
    print(f"  比例: {large_params/base_params:.2f}x")

    # 对比配置
    print(f"\n配置对比:")
    base_info = bert_base.get_model_info()
    large_info = bert_large.get_model_info()

    for key in ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size']:
        print(f"  {key}:")
        print(f"    Base:  {base_info[key]}")
        print(f"    Large: {large_info[key]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试 BERT-Large 模型")
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='跳过下载测试（假设模型已下载）'
    )
    parser.add_argument(
        '--comparison',
        action='store_true',
        help='运行 BERT-Base 和 BERT-Large 对比测试'
    )

    args = parser.parse_args()

    if not args.skip_download:
        success = test_bert_large_download_and_load()
        if not success:
            sys.exit(1)

    if args.comparison:
        test_bert_base_comparison()
