"""
GPT-2 Medium 使用示例。

展示如何加载和使用 GPT-2 Medium 模型进行各种任务。
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ckpt_compress.models.gpt2 import get_gpt2_medium
from src.ckpt_compress.utils.data_loader import get_wikitext2_dataloader


def example_1_basic_loading():
    """示例 1: 基本模型加载"""
    print("\n" + "=" * 60)
    print("示例 1: 基本模型加载")
    print("=" * 60)

    # 加载预训练模型
    model = get_gpt2_medium(pretrained=True)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型加载成功")
    print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    return model


def example_2_forward_pass():
    """示例 2: 前向传播"""
    print("\n" + "=" * 60)
    print("示例 2: 前向传播")
    print("=" * 60)

    model = get_gpt2_medium(pretrained=True)
    model.eval()

    # 创建随机输入
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_length))

    print(f"输入形状: {input_ids.shape}")

    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    print(f"输出 logits 形状: {logits.shape}")
    print(f"✓ 前向传播成功")


def example_3_compute_loss():
    """示例 3: 计算损失"""
    print("\n" + "=" * 60)
    print("示例 3: 计算损失（语言模型训练）")
    print("=" * 60)

    model = get_gpt2_medium(pretrained=True)
    model.train()

    # 准备数据
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_length))
    labels = input_ids.clone()

    # 前向传播（自动计算损失）
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    print(f"输入形状: {input_ids.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"✓ 损失计算成功")


def example_4_with_dataloader():
    """示例 4: 与数据加载器配合使用"""
    print("\n" + "=" * 60)
    print("示例 4: 与 WikiText-2 数据加载器配合使用")
    print("=" * 60)

    model = get_gpt2_medium(pretrained=True)
    model.eval()

    # 加载数据
    dataloader = get_wikitext2_dataloader(
        split='validation',
        batch_size=2,
        seq_length=128,
        max_samples=10,  # 仅使用 10 个样本进行演示
    )

    print(f"数据加载器创建成功，包含 {len(dataloader)} 个批次")

    # 计算平均损失
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids']
            labels = batch['labels']

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

            if i == 0:
                print(f"  批次 {i+1}: input_ids 形状 = {input_ids.shape}, loss = {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))

    print(f"\n平均损失: {avg_loss:.4f}")
    print(f"困惑度 (Perplexity): {perplexity.item():.2f}")
    print(f"✓ 数据加载器测试成功")


def example_5_save_and_load():
    """示例 5: 保存和加载模型"""
    print("\n" + "=" * 60)
    print("示例 5: 保存和加载模型状态字典")
    print("=" * 60)

    # 加载模型
    model = get_gpt2_medium(pretrained=True)

    # 保存状态字典
    save_path = "./data/models/gpt2_medium_example.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)
    print(f"✓ 状态字典已保存到: {save_path}")
    print(f"  包含 {len(state_dict)} 个参数")

    # 创建新模型并加载
    new_model = get_gpt2_medium(pretrained=False)  # 随机初始化
    new_model.load_state_dict(torch.load(save_path))
    print(f"✓ 状态字典已加载到新模型")

    # 验证一致性
    original_param = next(iter(model.parameters()))
    loaded_param = next(iter(new_model.parameters()))

    if torch.equal(original_param, loaded_param):
        print(f"✓ 参数一致性验证通过")
    else:
        print(f"✗ 参数不一致！")

    # 清理
    os.remove(save_path)
    print(f"✓ 临时文件已清理")


def example_6_parameter_inspection():
    """示例 6: 参数检查"""
    print("\n" + "=" * 60)
    print("示例 6: 检查模型参数")
    print("=" * 60)

    model = get_gpt2_medium(pretrained=True)

    # 统计不同类型的参数
    param_stats = {
        'embedding': 0,
        'attention': 0,
        'mlp': 0,
        'layernorm': 0,
        'other': 0,
    }

    print("\n前 10 个参数:")
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < 10:
            print(f"  {i+1}. {name}: {param.shape}")

        # 统计参数类型
        if 'wte' in name or 'wpe' in name:
            param_stats['embedding'] += param.numel()
        elif 'attn' in name:
            param_stats['attention'] += param.numel()
        elif 'mlp' in name:
            param_stats['mlp'] += param.numel()
        elif 'ln' in name:
            param_stats['layernorm'] += param.numel()
        else:
            param_stats['other'] += param.numel()

    print("\n参数统计:")
    total = sum(param_stats.values())
    for param_type, count in param_stats.items():
        percentage = (count / total) * 100
        print(f"  {param_type:12s}: {count:12,} ({percentage:5.2f}%)")
    print(f"  {'总计':12s}: {total:12,} (100.00%)")


def example_7_memory_efficient():
    """示例 7: 内存高效评估"""
    print("\n" + "=" * 60)
    print("示例 7: 内存高效评估（使用 MemoryEfficientEvaluator）")
    print("=" * 60)

    try:
        from src.ckpt_compress.methods.adam_prune.memory_efficient import (
            MemoryEfficientEvaluator
        )

        model = get_gpt2_medium(pretrained=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # 创建数据加载器
        dataloader = get_wikitext2_dataloader(
            split='train',
            batch_size=2,
            seq_length=128,
            max_samples=10,
        )

        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

        # 创建评估器
        evaluator = MemoryEfficientEvaluator(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            num_batches=5,
        )

        print(f"✓ MemoryEfficientEvaluator 创建成功")
        print(f"  设备: {device}")
        print(f"  批次数: 5")

        # 缓存批次
        evaluator.cache_batches()
        print(f"✓ 批次数据已缓存")

        # 计算基线损失
        baseline_loss = evaluator.compute_baseline_loss()
        print(f"✓ 基线损失: {baseline_loss:.4f}")

        print(f"\n提示: 使用 MemoryEfficientEvaluator 可以避免大模型的 deepcopy")
        print(f"      这对于 GPT-2 Medium 这样的大模型非常重要！")

    except ImportError as e:
        print(f"⚠️  MemoryEfficientEvaluator 不可用: {e}")
        print(f"   这是一个高级功能，用于 AdamPrune 实验")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("GPT-2 Medium 使用示例")
    print("=" * 60)

    # 运行所有示例
    example_1_basic_loading()
    example_2_forward_pass()
    example_3_compute_loss()
    example_4_with_dataloader()
    example_5_save_and_load()
    example_6_parameter_inspection()
    example_7_memory_efficient()

    # 总结
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)
    print("\n更多信息请参考:")
    print("  - docs/GPT2_MEDIUM_GUIDE.md")
    print("  - PROJECT_ANALYSIS.md")
    print("  - CLAUDE.md")


if __name__ == '__main__':
    main()
