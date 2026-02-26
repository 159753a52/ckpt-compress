"""
加载微调后的 GPT-2 检查点示例脚本。

演示如何加载和使用微调后的模型。
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small


def load_checkpoint(checkpoint_path: str):
    """
    加载检查点。

    参数:
        checkpoint_path: 检查点文件路径

    返回:
        model: 加载了权重的模型
        optimizer: 加载了状态的优化器
        step: 训练步数
    """
    print(f"加载检查点: {checkpoint_path}")

    # 创建模型
    model = get_gpt2_small(pretrained=False)

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']

    print(f"✓ 成功加载检查点 (步数: {step})")

    return model, optimizer, step


def print_model_info(model):
    """打印模型信息。"""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型信息:")
    print(f"  总参数量: {num_params:,}")
    print(f"  可训练参数: {num_trainable:,}")
    print(f"  参数大小: {num_params * 4 / 1024 / 1024:.2f} MB (float32)")


def generate_text_example(model, device='cpu'):
    """
    文本生成示例。

    注意: 这只是一个简单示例，实际生成需要使用 transformers 的 generate() 方法。
    """
    try:
        from transformers import GPT2Tokenizer
    except ImportError:
        print("\n警告: transformers 库未安装，跳过文本生成示例")
        return

    print(f"\n文本生成示例:")
    print("=" * 60)

    # 加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 准备输入
    input_text = "The quick brown fox"
    print(f"输入文本: {input_text}")

    # 编码
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # 前向传播
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        logits = model(input_ids)

    # 获取下一个词的预测
    next_token_logits = logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    next_token = tokenizer.decode([next_token_id])

    print(f"预测的下一个词: '{next_token}'")
    print(f"Logits 形状: {logits.shape}")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='加载微调后的 GPT-2 检查点')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt',
                        help='检查点文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='设备')
    parser.add_argument('--generate', action='store_true',
                        help='运行文本生成示例')

    args = parser.parse_args()

    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'

    print("=" * 60)
    print("加载微调后的 GPT-2 Small 检查点")
    print("=" * 60)

    # 加载检查点
    model, optimizer, step = load_checkpoint(args.checkpoint)

    # 打印模型信息
    print_model_info(model)

    # 打印优化器信息
    print(f"\n优化器信息:")
    print(f"  类型: {type(optimizer).__name__}")
    print(f"  参数组数: {len(optimizer.param_groups)}")
    print(f"  学习率: {optimizer.param_groups[0]['lr']}")
    print(f"  权重衰减: {optimizer.param_groups[0]['weight_decay']}")

    # 文本生成示例
    if args.generate:
        generate_text_example(model, device)

    print("\n" + "=" * 60)
    print("检查点加载完成！")
    print("=" * 60)

    # 使用提示
    print("\n使用提示:")
    print("1. 继续训练:")
    print("   model.train()")
    print("   # 继续训练循环...")
    print()
    print("2. 推理:")
    print("   model.eval()")
    print("   with torch.no_grad():")
    print("       logits = model(input_ids)")
    print()
    print("3. 压缩检查点:")
    print("   python experiments/scripts/compress_and_resume.py \\")
    print("     --mode compress \\")
    print(f"     --checkpoint {args.checkpoint} \\")
    print("     --method excp \\")
    print("     --output compressed/model_excp.bin")


if __name__ == '__main__':
    main()
