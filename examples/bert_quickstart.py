"""
BERT-Large 快速开始示例。

这个脚本展示了如何快速开始使用 BERT-Large 模型。
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def example_1_basic_usage():
    """示例 1: 基本使用"""
    print("\n" + "="*60)
    print("示例 1: BERT-Large 基本使用")
    print("="*60)

    from src.ckpt_compress.models.bert import get_bert_large

    # 加载预训练模型
    print("\n加载 BERT-Large 预训练模型...")
    model = get_bert_large(pretrained=True, cache_dir="./data/models")
    model.eval()

    # 创建虚拟输入
    batch_size = 2
    seq_length = 64
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    print(f"输入形状: {input_ids.shape}")

    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    print(f"输出形状: {logits.shape}")
    print(f"输出范围: [{logits.min():.2f}, {logits.max():.2f}]")
    print("✓ 基本使用成功！")


def example_2_model_info():
    """示例 2: 查看模型信息"""
    print("\n" + "="*60)
    print("示例 2: 查看 BERT-Large 模型信息")
    print("="*60)

    from src.ckpt_compress.models.bert import get_bert_large

    # 加载模型（使用本地缓存）
    model = get_bert_large(
        pretrained=True,
        local_files_only=True,
        cache_dir="./data/models"
    )

    # 获取模型配置
    print("\n模型配置:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 计算参数量
    print("\n参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    # 查看状态字典
    print("\n状态字典:")
    state_dict = model.state_dict()
    print(f"  包含 {len(state_dict)} 个张量")
    print(f"  前 3 个键:")
    for key in list(state_dict.keys())[:3]:
        tensor = state_dict[key]
        print(f"    - {key}: {tensor.shape}")


def example_3_compare_models():
    """示例 3: 对比 BERT-Base 和 BERT-Large"""
    print("\n" + "="*60)
    print("示例 3: 对比 BERT-Base 和 BERT-Large")
    print("="*60)

    from src.ckpt_compress.models.bert import get_bert_base, get_bert_large

    # 加载两个模型（随机初始化以节省时间）
    print("\n加载模型（随机初始化）...")
    bert_base = get_bert_base(pretrained=False)
    bert_large = get_bert_large(pretrained=False)

    # 对比参数量
    base_params = sum(p.numel() for p in bert_base.parameters())
    large_params = sum(p.numel() for p in bert_large.parameters())

    print(f"\n参数量对比:")
    print(f"  BERT-Base:  {base_params:,} ({base_params/1e6:.1f}M)")
    print(f"  BERT-Large: {large_params:,} ({large_params/1e6:.1f}M)")
    print(f"  BERT-Large 是 BERT-Base 的 {large_params/base_params:.2f} 倍")

    # 对比配置
    print(f"\n配置对比:")
    base_info = bert_base.get_model_info()
    large_info = bert_large.get_model_info()

    print(f"  {'配置项':<25} {'BERT-Base':<15} {'BERT-Large':<15}")
    print(f"  {'-'*55}")
    for key in ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size']:
        print(f"  {key:<25} {base_info[key]:<15} {large_info[key]:<15}")


def example_4_state_dict_operations():
    """示例 4: 状态字典操作"""
    print("\n" + "="*60)
    print("示例 4: 状态字典保存和加载")
    print("="*60)

    from src.ckpt_compress.models.bert import get_bert_large
    import tempfile

    # 加载模型
    print("\n加载模型...")
    model = get_bert_large(pretrained=False)

    # 保存状态字典
    print("\n保存状态字典...")
    state_dict = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
        torch.save(state_dict, temp_path)
        file_size = os.path.getsize(temp_path)
        print(f"  保存到: {temp_path}")
        print(f"  文件大小: {file_size / 1024 / 1024:.2f} MB")

    # 加载状态字典
    print("\n加载状态字典...")
    loaded_state_dict = torch.load(temp_path)
    print(f"  加载了 {len(loaded_state_dict)} 个张量")

    # 创建新模型并加载权重
    print("\n创建新模型并加载权重...")
    new_model = get_bert_large(pretrained=False)
    new_model.load_state_dict(loaded_state_dict)
    print("  ✓ 权重加载成功！")

    # 验证权重一致性
    print("\n验证权重一致性...")
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(),
        new_model.named_parameters()
    ):
        assert name1 == name2
        assert torch.allclose(param1, param2)
    print("  ✓ 所有权重一致！")

    # 清理临时文件
    os.unlink(temp_path)


def main():
    """运行所有示例"""
    import argparse

    parser = argparse.ArgumentParser(description="BERT-Large 快速开始示例")
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4],
        help='运行指定的示例 (1-4)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='运行所有示例'
    )

    args = parser.parse_args()

    examples = {
        1: example_1_basic_usage,
        2: example_2_model_info,
        3: example_3_compare_models,
        4: example_4_state_dict_operations,
    }

    if args.all:
        print("\n" + "="*60)
        print("运行所有 BERT-Large 示例")
        print("="*60)
        for i in range(1, 5):
            try:
                examples[i]()
            except Exception as e:
                print(f"\n✗ 示例 {i} 失败: {e}")
    elif args.example:
        try:
            examples[args.example]()
        except Exception as e:
            print(f"\n✗ 示例 {args.example} 失败: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        print("\n提示: 使用 --all 运行所有示例，或使用 --example N 运行指定示例")


if __name__ == "__main__":
    main()
