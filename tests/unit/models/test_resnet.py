"""
ResNet 模型测试。

TDD：先写测试，再实现。
测试覆盖：ResNet-18 和 ResNet-50 的所有变体。
"""

import pytest
import torch

from ckpt_compress.models.resnet import (
    ResNet18ForCIFAR10,
    ResNet50ForCIFAR,
    ResNet50ForImageNet,
    get_resnet18_cifar10,
    get_resnet50_cifar100,
    get_resnet50_tiny_imagenet,
    get_resnet50_imagenet,
)


class TestResNet18ForCIFAR10:
    """适配 CIFAR-10 的 ResNet18 测试。"""

    def test_model_creation(self):
        """模型应成功创建。"""
        model = ResNet18ForCIFAR10()
        assert model is not None

    def test_model_output_shape(self):
        """CIFAR-10 的输出形状应为 (batch_size, 10)。"""
        model = ResNet18ForCIFAR10()
        x = torch.randn(4, 3, 32, 32)  # CIFAR-10 输入尺寸

        output = model(x)

        assert output.shape == (4, 10)

    def test_model_accepts_cifar10_input(self):
        """模型应接受 32x32 RGB 图像。"""
        model = ResNet18ForCIFAR10()
        x = torch.randn(1, 3, 32, 32)

        # 不应抛出异常
        output = model(x)
        assert output.shape == (1, 10)

    def test_model_parameter_count(self):
        """CIFAR-10 的 ResNet18 应有约 11M 参数。"""
        model = ResNet18ForCIFAR10()

        param_count = sum(p.numel() for p in model.parameters())

        # ResNet18 有约 11M 参数
        assert 10_000_000 < param_count < 15_000_000

    def test_model_trainable(self):
        """模型应可训练（梯度流动）。"""
        model = ResNet18ForCIFAR10()
        x = torch.randn(2, 3, 32, 32)
        y = torch.tensor([0, 1])

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()

        # 检查梯度存在
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_model_eval_mode(self):
        """模型应在评估模式下工作。"""
        model = ResNet18ForCIFAR10()
        model.eval()
        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 10)

    def test_named_parameters(self):
        """模型应返回命名参数。"""
        model = ResNet18ForCIFAR10()
        named_params = list(model.named_parameters())

        assert len(named_params) > 0
        # 检查返回的是 (name, param) 元组
        assert isinstance(named_params[0], tuple)
        assert isinstance(named_params[0][0], str)
        assert isinstance(named_params[0][1], torch.nn.Parameter)

    def test_parameters_method(self):
        """模型应返回参数迭代器。"""
        model = ResNet18ForCIFAR10()
        params = list(model.parameters())

        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)


class TestGetResNet18CIFAR10:
    """工厂函数测试。"""

    def test_get_model_default(self):
        """工厂应返回默认设置的模型。"""
        model = get_resnet18_cifar10()

        assert isinstance(model, ResNet18ForCIFAR10)

    def test_get_model_pretrained_false(self):
        """工厂应返回随机初始化的模型。"""
        model = get_resnet18_cifar10(pretrained=False)

        assert isinstance(model, ResNet18ForCIFAR10)

    def test_get_model_num_classes(self):
        """工厂应支持自定义类别数。"""
        model = get_resnet18_cifar10(num_classes=100)
        x = torch.randn(1, 3, 32, 32)

        output = model(x)

        assert output.shape == (1, 100)


# ============================================================
# ResNet-50 for CIFAR 测试
# ============================================================

class TestResNet50ForCIFAR:
    """ResNet50 for CIFAR 测试。"""

    def test_model_creation_cifar100(self):
        """模型应成功创建（CIFAR-100）。"""
        model = ResNet50ForCIFAR(num_classes=100)
        assert model is not None

    def test_model_creation_cifar10(self):
        """模型应成功创建（CIFAR-10）。"""
        model = ResNet50ForCIFAR(num_classes=10)
        assert model is not None

    def test_model_output_shape_cifar100(self):
        """CIFAR-100 的输出形状应为 (batch_size, 100)。"""
        model = ResNet50ForCIFAR(num_classes=100)
        x = torch.randn(4, 3, 32, 32)

        output = model(x)

        assert output.shape == (4, 100)

    def test_model_output_shape_cifar10(self):
        """CIFAR-10 的输出形状应为 (batch_size, 10)。"""
        model = ResNet50ForCIFAR(num_classes=10)
        x = torch.randn(4, 3, 32, 32)

        output = model(x)

        assert output.shape == (4, 10)

    def test_model_accepts_cifar_input(self):
        """模型应接受 32x32 RGB 图像。"""
        model = ResNet50ForCIFAR(num_classes=100)
        x = torch.randn(1, 3, 32, 32)

        output = model(x)
        assert output.shape == (1, 100)

    def test_model_parameter_count(self):
        """ResNet50 应有约 25M 参数。"""
        model = ResNet50ForCIFAR(num_classes=100)

        param_count = sum(p.numel() for p in model.parameters())

        # ResNet50 有约 25M 参数
        assert 20_000_000 < param_count < 30_000_000

    def test_model_trainable(self):
        """模型应可训练（梯度流动）。"""
        model = ResNet50ForCIFAR(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        y = torch.tensor([0, 1])

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()

        # 检查梯度存在
        has_grad = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_model_eval_mode(self):
        """模型应在评估模式下工作。"""
        model = ResNet50ForCIFAR(num_classes=100)
        model.eval()
        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 100)

    def test_model_train_mode(self):
        """模型应在训练模式下工作。"""
        model = ResNet50ForCIFAR(num_classes=100)
        model.train()
        x = torch.randn(2, 3, 32, 32)

        output = model(x)

        assert output.shape == (2, 100)

    def test_state_dict_save_load(self):
        """模型应支持状态字典保存和加载。"""
        model1 = ResNet50ForCIFAR(num_classes=100)
        model2 = ResNet50ForCIFAR(num_classes=100)

        # 保存模型1的状态
        state_dict = model1.state_dict()

        # 加载到模型2
        model2.load_state_dict(state_dict)

        # 验证参数相同
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_named_parameters(self):
        """模型应返回命名参数。"""
        model = ResNet50ForCIFAR(num_classes=100)
        named_params = list(model.named_parameters())

        assert len(named_params) > 0
        assert isinstance(named_params[0], tuple)

    def test_parameters_method(self):
        """模型应返回参数迭代器。"""
        model = ResNet50ForCIFAR(num_classes=100)
        params = list(model.parameters())

        assert len(params) > 0


# ============================================================
# ResNet-50 for ImageNet 测试
# ============================================================

class TestResNet50ForImageNet:
    """ResNet50 for ImageNet 测试。"""

    def test_model_creation_imagenet(self):
        """模型应成功创建（ImageNet）。"""
        model = ResNet50ForImageNet(num_classes=1000, pretrained=False)
        assert model is not None

    def test_model_creation_tiny_imagenet(self):
        """模型应成功创建（Tiny-ImageNet）。"""
        model = ResNet50ForImageNet(num_classes=200, pretrained=False)
        assert model is not None

    def test_model_output_shape_imagenet(self):
        """ImageNet 的输出形状应为 (batch_size, 1000)。"""
        model = ResNet50ForImageNet(num_classes=1000, pretrained=False)
        x = torch.randn(4, 3, 224, 224)

        output = model(x)

        assert output.shape == (4, 1000)

    def test_model_output_shape_tiny_imagenet(self):
        """Tiny-ImageNet 的输出形状应为 (batch_size, 200)。"""
        model = ResNet50ForImageNet(num_classes=200, pretrained=False)
        x = torch.randn(4, 3, 64, 64)

        output = model(x)

        assert output.shape == (4, 200)

    def test_model_accepts_imagenet_input(self):
        """模型应接受 224x224 RGB 图像。"""
        model = ResNet50ForImageNet(num_classes=1000, pretrained=False)
        x = torch.randn(1, 3, 224, 224)

        output = model(x)
        assert output.shape == (1, 1000)

    def test_model_parameter_count(self):
        """ResNet50 应有约 25M 参数。"""
        model = ResNet50ForImageNet(num_classes=1000, pretrained=False)

        param_count = sum(p.numel() for p in model.parameters())

        # ResNet50 有约 25M 参数
        assert 20_000_000 < param_count < 30_000_000

    def test_model_trainable(self):
        """模型应可训练（梯度流动）。"""
        model = ResNet50ForImageNet(num_classes=1000, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        y = torch.tensor([0, 1])

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        loss.backward()

        # 检查梯度存在
        has_grad = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_model_eval_mode(self):
        """模型应在评估模式下工作。"""
        model = ResNet50ForImageNet(num_classes=1000, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 1000)

    def test_state_dict_save_load(self):
        """模型应支持状态字典保存和加载。"""
        model1 = ResNet50ForImageNet(num_classes=1000, pretrained=False)
        model2 = ResNet50ForImageNet(num_classes=1000, pretrained=False)

        # 保存模型1的状态
        state_dict = model1.state_dict()

        # 加载到模型2
        model2.load_state_dict(state_dict)

        # 验证参数相同
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_named_parameters(self):
        """模型应返回命名参数。"""
        model = ResNet50ForImageNet(num_classes=1000, pretrained=False)
        named_params = list(model.named_parameters())

        assert len(named_params) > 0
        assert isinstance(named_params[0], tuple)

    def test_parameters_method(self):
        """模型应返回参数迭代器。"""
        model = ResNet50ForImageNet(num_classes=1000, pretrained=False)
        params = list(model.parameters())

        assert len(params) > 0

    def test_train_mode(self):
        """模型应在训练模式下工作。"""
        model = ResNet50ForImageNet(num_classes=1000, pretrained=False)
        model.train()
        x = torch.randn(2, 3, 224, 224)

        output = model(x)

        assert output.shape == (2, 1000)


# ============================================================
# 工厂函数测试
# ============================================================

class TestResNet50FactoryFunctions:
    """ResNet50 工厂函数测试。"""

    def test_get_resnet50_cifar100(self):
        """工厂应返回 CIFAR-100 模型。"""
        model = get_resnet50_cifar100()

        assert isinstance(model, ResNet50ForCIFAR)
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 100)

    def test_get_resnet50_tiny_imagenet(self):
        """工厂应返回 Tiny-ImageNet 模型。"""
        model = get_resnet50_tiny_imagenet(pretrained=False)

        assert isinstance(model, ResNet50ForImageNet)
        x = torch.randn(1, 3, 64, 64)
        output = model(x)
        assert output.shape == (1, 200)

    def test_get_resnet50_imagenet(self):
        """工厂应返回 ImageNet 模型。"""
        model = get_resnet50_imagenet(pretrained=False)

        assert isinstance(model, ResNet50ForImageNet)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 1000)
