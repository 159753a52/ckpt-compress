"""
GPT-2 模型工具测试。

TDD：先写测试，再实现。
"""

import pytest
import torch
import sys
from unittest.mock import patch

from ckpt_compress.models.gpt2 import (
    GPT2ForExperiment,
    GPT2MediumForExperiment,
    get_gpt2_small,
    get_gpt2_medium,
)


class TestGPT2ForExperiment:
    """GPT-2 实验模型测试。"""

    def test_model_creation(self):
        """模型应成功创建。"""
        model = GPT2ForExperiment()
        assert model is not None

    def test_model_output_shape(self):
        """输出形状应为 (batch_size, seq_len, vocab_size)。"""
        model = GPT2ForExperiment()
        # GPT-2 输入是 token ids
        input_ids = torch.randint(0, 50257, (2, 16))  # batch=2, seq_len=16

        output = model(input_ids)

        # 输出应该是 logits
        assert output.shape == (2, 16, 50257)  # vocab_size=50257

    def test_model_accepts_variable_length(self):
        """模型应接受不同长度的序列。"""
        model = GPT2ForExperiment()

        # 短序列
        short_input = torch.randint(0, 50257, (1, 8))
        short_output = model(short_input)
        assert short_output.shape == (1, 8, 50257)

        # 长序列
        long_input = torch.randint(0, 50257, (1, 64))
        long_output = model(long_input)
        assert long_output.shape == (1, 64, 50257)

    def test_model_parameter_count(self):
        """GPT-2 small 应有约 124M 参数。"""
        model = GPT2ForExperiment()

        param_count = sum(p.numel() for p in model.parameters())

        # GPT-2 small 有约 124M 参数
        assert 100_000_000 < param_count < 150_000_000

    def test_model_trainable(self):
        """模型应可训练（梯度流动）。"""
        model = GPT2ForExperiment()
        input_ids = torch.randint(0, 50257, (2, 16))

        output = model(input_ids)
        # 使用简单的损失函数
        loss = output.mean()
        loss.backward()

        # 检查梯度存在
        grad_count = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1

        assert grad_count > 0

    def test_model_eval_mode(self):
        """模型应在评估模式下工作。"""
        model = GPT2ForExperiment()
        model.eval()
        input_ids = torch.randint(0, 50257, (2, 16))

        with torch.no_grad():
            output = model(input_ids)

        assert output.shape == (2, 16, 50257)

    def test_model_has_transformer_layers(self):
        """模型应包含 Transformer 层。"""
        model = GPT2ForExperiment()

        # 检查是否有 attention 相关参数
        param_names = [name for name, _ in model.named_parameters()]

        # GPT-2 应该有 attention 层
        has_attention = any('attn' in name.lower() for name in param_names)
        assert has_attention, "模型应包含 attention 层"

    def test_model_layer_types(self):
        """模型应包含典型的 Transformer 组件。"""
        model = GPT2ForExperiment()
        param_names = [name for name, _ in model.named_parameters()]

        # 检查关键组件
        has_embedding = any('wte' in name or 'embed' in name.lower() for name in param_names)
        has_ln = any('ln' in name.lower() for name in param_names)

        assert has_embedding, "模型应包含 embedding 层"
        assert has_ln, "模型应包含 layer norm"

    def test_state_dict_method(self):
        """模型应支持 state_dict() 方法。"""
        model = GPT2ForExperiment()
        state_dict = model.state_dict()

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    def test_load_state_dict_method(self):
        """模型应支持 load_state_dict() 方法。"""
        model1 = GPT2ForExperiment()
        model2 = GPT2ForExperiment()

        # 保存模型1的状态
        state_dict = model1.state_dict()

        # 加载到模型2
        model2.load_state_dict(state_dict)

        # 验证参数相同
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_train_method(self):
        """模型应支持 train() 方法切换到训练模式。"""
        model = GPT2ForExperiment()
        model.eval()  # 先设置为评估模式

        # 切换到训练模式
        result = model.train()

        # train() 应该返回 self
        assert result is model

        # 验证模型在训练模式
        assert model.model.training is True

    def test_train_mode_false(self):
        """模型应支持 train(False) 切换到评估模式。"""
        model = GPT2ForExperiment()
        model.train()  # 先设置为训练模式

        # 切换到评估模式
        result = model.train(mode=False)

        # train(False) 应该返回 self
        assert result is model

        # 验证模型在评估模式
        assert model.model.training is False


class TestGetGPT2Small:
    """工厂函数测试。"""

    def test_get_model_default(self):
        """工厂应返回默认设置的模型。"""
        model = get_gpt2_small()

        assert isinstance(model, GPT2ForExperiment)

    def test_get_model_pretrained_false(self):
        """工厂应支持随机初始化。"""
        model = get_gpt2_small(pretrained=False)

        assert isinstance(model, GPT2ForExperiment)

    def test_get_model_pretrained_true(self):
        """工厂应支持加载预训练权重。"""
        # 注意：这个测试可能需要下载模型，可能会慢
        model = get_gpt2_small(pretrained=True)

        assert isinstance(model, GPT2ForExperiment)


class TestGPT2ForAdamPrune:
    """GPT-2 用于 AdamPrune 实验的测试。"""

    def test_can_extract_gradients(self):
        """应能提取梯度用于重要性计算。"""
        model = GPT2ForExperiment()
        input_ids = torch.randint(0, 50257, (2, 16))

        output = model(input_ids)
        loss = output.mean()
        loss.backward()

        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        assert len(gradients) > 0

    def test_can_use_with_adam_optimizer(self):
        """应能与 Adam 优化器配合使用。"""
        model = GPT2ForExperiment()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        input_ids = torch.randint(0, 50257, (2, 16))
        output = model(input_ids)
        loss = output.mean()
        loss.backward()
        optimizer.step()

        # 检查优化器状态
        has_exp_avg_sq = False
        for param in model.parameters():
            if param in optimizer.state:
                if 'exp_avg_sq' in optimizer.state[param]:
                    has_exp_avg_sq = True
                    break

        assert has_exp_avg_sq, "Adam 优化器应有 exp_avg_sq 状态"

    def test_parameter_distribution_by_layer_type(self):
        """应能按层类型统计参数分布。"""
        model = GPT2ForExperiment()

        layer_stats = {
            'embedding': 0,
            'attention': 0,
            'mlp': 0,
            'layernorm': 0,
            'other': 0,
        }

        for name, param in model.named_parameters():
            name_lower = name.lower()
            if 'wte' in name_lower or 'wpe' in name_lower or 'embed' in name_lower:
                layer_stats['embedding'] += param.numel()
            elif 'attn' in name_lower:
                layer_stats['attention'] += param.numel()
            elif 'mlp' in name_lower or 'fc' in name_lower:
                layer_stats['mlp'] += param.numel()
            elif 'ln' in name_lower or 'norm' in name_lower:
                layer_stats['layernorm'] += param.numel()
            else:
                layer_stats['other'] += param.numel()

        # GPT-2 应该有大量参数在 attention 和 mlp 层
        total = sum(layer_stats.values())
        assert layer_stats['attention'] > 0, "应有 attention 参数"
        assert layer_stats['mlp'] > 0, "应有 MLP 参数"
        assert layer_stats['embedding'] > 0, "应有 embedding 参数"


# ============================================================
# GPT-2 Medium 测试
# ============================================================

class TestGPT2MediumForExperiment:
    """GPT-2 Medium 实验模型测试。"""

    def test_model_creation(self):
        """模型应成功创建。"""
        model = GPT2MediumForExperiment(pretrained=False)
        assert model is not None

    def test_model_output_shape(self):
        """输出形状应为 (batch_size, seq_len, vocab_size)。"""
        model = GPT2MediumForExperiment(pretrained=False)
        input_ids = torch.randint(0, 50257, (2, 16))

        output = model(input_ids)

        assert output.logits.shape == (2, 16, 50257)

    def test_model_parameter_count(self):
        """GPT-2 medium 应有约 355M 参数。"""
        model = GPT2MediumForExperiment(pretrained=False)

        param_count = sum(p.numel() for p in model.parameters())

        # GPT-2 medium 有约 355M 参数
        assert 300_000_000 < param_count < 400_000_000

    def test_model_has_24_layers(self):
        """GPT-2 medium 应有 24 层。"""
        model = GPT2MediumForExperiment(pretrained=False)

        # 检查配置
        assert model.model.config.n_layer == 24

    def test_model_hidden_size(self):
        """GPT-2 medium 应有 1024 隐藏维度。"""
        model = GPT2MediumForExperiment(pretrained=False)

        assert model.model.config.n_embd == 1024

    def test_model_trainable(self):
        """模型应可训练（梯度流动）。"""
        model = GPT2MediumForExperiment(pretrained=False)
        input_ids = torch.randint(0, 50257, (2, 16))

        output = model(input_ids)
        loss = output.logits.mean()
        loss.backward()

        # 检查梯度存在
        grad_count = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1

        assert grad_count > 0

    def test_model_eval_mode(self):
        """模型应在评估模式下工作。"""
        model = GPT2MediumForExperiment(pretrained=False)
        model.eval()
        input_ids = torch.randint(0, 50257, (2, 16))

        with torch.no_grad():
            output = model(input_ids)

        assert output.logits.shape == (2, 16, 50257)

    def test_model_train_mode(self):
        """模型应在训练模式下工作。"""
        model = GPT2MediumForExperiment(pretrained=False)
        model.train()
        input_ids = torch.randint(0, 50257, (2, 16))

        output = model(input_ids)

        assert output.logits.shape == (2, 16, 50257)

    def test_forward_with_labels(self):
        """模型应支持带标签的前向传播（计算损失）。"""
        model = GPT2MediumForExperiment(pretrained=False)
        input_ids = torch.randint(0, 50257, (2, 16))
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)

        assert hasattr(output, 'loss')
        assert output.loss is not None

    def test_state_dict_save_load(self):
        """模型应支持状态字典保存和加载。"""
        model1 = GPT2MediumForExperiment(pretrained=False)
        model2 = GPT2MediumForExperiment(pretrained=False)

        # 保存模型1的状态
        state_dict = model1.state_dict()

        # 加载到模型2
        model2.load_state_dict(state_dict)

        # 验证参数相同
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_named_parameters(self):
        """模型应返回命名参数。"""
        model = GPT2MediumForExperiment(pretrained=False)
        named_params = list(model.named_parameters())

        assert len(named_params) > 0
        assert isinstance(named_params[0], tuple)

    def test_parameters_method(self):
        """模型应返回参数迭代器。"""
        model = GPT2MediumForExperiment(pretrained=False)
        params = list(model.parameters())

        assert len(params) > 0


class TestGetGPT2Medium:
    """GPT-2 Medium 工厂函数测试。"""

    def test_get_model_default(self):
        """工厂应返回默认设置的模型。"""
        model = get_gpt2_medium(pretrained=False)

        assert isinstance(model, GPT2MediumForExperiment)

    def test_get_model_pretrained_false(self):
        """工厂应支持随机初始化。"""
        model = get_gpt2_medium(pretrained=False)

        assert isinstance(model, GPT2MediumForExperiment)
        # 验证参数量
        param_count = sum(p.numel() for p in model.parameters())
        assert 300_000_000 < param_count < 400_000_000

    @pytest.mark.slow
    def test_get_model_pretrained_true(self):
        """工厂应支持加载预训练权重。"""
        # 注意：这个测试需要下载模型，可能会慢
        model = get_gpt2_medium(pretrained=True)

        assert isinstance(model, GPT2MediumForExperiment)
        # 验证参数量
        param_count = sum(p.numel() for p in model.parameters())
        assert 300_000_000 < param_count < 400_000_000


class TestImportError:
    """测试 transformers 库未安装时的异常处理。"""

    def test_gpt2_import_error(self):
        """当 transformers 未安装时，应抛出 ImportError。"""
        # 模拟 transformers 未安装
        with patch.dict(sys.modules, {'transformers': None}):
            # 重新导入模块以触发 ImportError 检查
            import importlib
            import ckpt_compress.models.gpt2 as gpt2_module

            # 重新加载模块
            importlib.reload(gpt2_module)

            # 尝试创建模型应该抛出 ImportError
            with pytest.raises(ImportError, match="transformers 库未安装"):
                gpt2_module.GPT2ForExperiment()

    def test_gpt2_medium_import_error(self):
        """当 transformers 未安装时，GPT2Medium 应抛出 ImportError。"""
        # 模拟 transformers 未安装
        with patch.dict(sys.modules, {'transformers': None}):
            # 重新导入模块以触发 ImportError 检查
            import importlib
            import ckpt_compress.models.gpt2 as gpt2_module

            # 重新加载模块
            importlib.reload(gpt2_module)

            # 尝试创建模型应该抛出 ImportError
            with pytest.raises(ImportError, match="transformers 库未安装"):
                gpt2_module.GPT2MediumForExperiment()
