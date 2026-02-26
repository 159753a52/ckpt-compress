"""
测试 GPT-2 Medium 模型加载和推理。

验证预训练模型可以正确加载并进行前向传播。
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ckpt_compress.models.gpt2 import get_gpt2_medium


def test_model_loading():
    """测试模型加载。"""
    print("=" * 60)
    print("测试 1: 加载预训练 GPT-2 Medium 模型")
    print("=" * 60)

    try:
        # 加载预训练模型
        model = get_gpt2_medium(pretrained=True)
        print("✓ 模型加载成功！")

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n模型信息:")
        print(f"  - 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  - 可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

        return model

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None


def test_forward_pass(model):
    """测试前向传播。"""
    print("\n" + "=" * 60)
    print("测试 2: 前向传播")
    print("=" * 60)

    try:
        # 创建测试输入
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 50257, (batch_size, seq_length))

        print(f"输入形状: {input_ids.shape}")

        # 前向传播
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        print(f"输出 logits 形状: {logits.shape}")
        print(f"预期形状: ({batch_size}, {seq_length}, 50257)")

        # 验证输出形状
        assert logits.shape == (batch_size, seq_length, 50257), "输出形状不正确！"
        print("✓ 前向传播成功！")

        return True

    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False


def test_text_generation(model):
    """测试文本生成。"""
    print("\n" + "=" * 60)
    print("测试 3: 文本生成")
    print("=" * 60)

    try:
        from transformers import GPT2Tokenizer

        # 加载 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

        # 测试文本
        prompt = "The future of artificial intelligence is"
        print(f"输入提示: '{prompt}'")

        # 编码
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # 生成
        model.eval()
        with torch.no_grad():
            outputs = model.model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )

        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n生成文本:\n{generated_text}")
        print("\n✓ 文本生成成功！")

        return True

    except Exception as e:
        print(f"✗ 文本生成失败: {e}")
        return False


def test_state_dict():
    """测试状态字典保存和加载。"""
    print("\n" + "=" * 60)
    print("测试 4: 状态字典保存和加载")
    print("=" * 60)

    try:
        # 加载模型
        model = get_gpt2_medium(pretrained=True)

        # 获取状态字典
        state_dict = model.state_dict()
        print(f"状态字典包含 {len(state_dict)} 个键")

        # 显示前 5 个键
        print("\n前 5 个参数键:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            shape = state_dict[key].shape
            print(f"  {i+1}. {key}: {shape}")

        # 保存到临时文件
        temp_path = "./data/models/test_gpt2_medium.pt"
        torch.save(state_dict, temp_path)
        print(f"\n✓ 状态字典已保存到: {temp_path}")

        # 加载回来
        loaded_state_dict = torch.load(temp_path)
        print(f"✓ 状态字典已加载，包含 {len(loaded_state_dict)} 个键")

        # 验证一致性
        for key in state_dict.keys():
            if not torch.equal(state_dict[key], loaded_state_dict[key]):
                print(f"✗ 键 {key} 的值不一致！")
                return False

        print("✓ 状态字典保存和加载成功！")

        # 清理临时文件
        os.remove(temp_path)

        return True

    except Exception as e:
        print(f"✗ 状态字典测试失败: {e}")
        return False


def main():
    """主函数。"""
    print("\n" + "=" * 60)
    print("GPT-2 Medium 模型测试")
    print("=" * 60 + "\n")

    # 测试 1: 加载模型
    model = test_model_loading()
    if model is None:
        print("\n❌ 测试失败：无法加载模型")
        return

    # 测试 2: 前向传播
    if not test_forward_pass(model):
        print("\n❌ 测试失败：前向传播错误")
        return

    # 测试 3: 文本生成
    test_text_generation(model)

    # 测试 4: 状态字典
    if not test_state_dict():
        print("\n❌ 测试失败：状态字典错误")
        return

    # 总结
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    print("\nGPT-2 Medium 模型已成功下载并可以正常使用。")
    print(f"模型缓存位置: ./data/models/models--gpt2-medium/")
    print("\n使用方法:")
    print("  from src.ckpt_compress.models.gpt2 import get_gpt2_medium")
    print("  model = get_gpt2_medium(pretrained=True)")


if __name__ == '__main__':
    main()
