"""
适用于 CIFAR-10 的 ResNet 模型。

CIFAR-10 图像为 32x32，因此需要修改标准 ResNet:
1. 第一个卷积: 使用 3x3 卷积核而非 7x7，步长 1 而非 2
2. 移除初始最大池化层
3. 输出 10 个类别而非 1000 个
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

from ._wrapper import DelegatingModel


class _ResNetWrapper(DelegatingModel):
    """Apply the shared wrapper contract to the legacy ``resnet`` attribute."""

    @property
    def _wrapped_model(self) -> nn.Module:
        return self.resnet


class ResNet18ForCIFAR10(_ResNetWrapper):
    """
    适用于 CIFAR-10 的 ResNet18（32x32 图像，10 个类别）。

    相比标准 ResNet18 的修改:
    - 第一个卷积: 3x3, stride=1, padding=1（而非 7x7, stride=2）
    - 第一个卷积后无最大池化
    - 最终全连接层输出 10 个类别（或自定义 num_classes）
    """

    def __init__(self, num_classes: int = 10):
        """
        初始化适用于 CIFAR-10 的 ResNet18。

        参数:
            num_classes: 输出类别数（CIFAR-10 默认为 10）
        """
        super().__init__()

        # 加载标准 ResNet18 架构
        self.resnet = resnet18(weights=None)

        # 修改第一个卷积层以适应 32x32 输入
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 移除最大池化（替换为恒等映射）
        self.resnet.maxpool = nn.Identity()

        # 修改最终全连接层以适应 num_classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 形状为 (batch_size, 3, 32, 32) 的输入张量

        返回:
            形状为 (batch_size, num_classes) 的输出张量
        """
        return self.resnet(x)

def get_resnet18_cifar10(
    num_classes: int = 10,
    pretrained: bool = False
) -> ResNet18ForCIFAR10:
    """
    创建适用于 CIFAR-10 的 ResNet18 的工厂函数。

    参数:
        num_classes: 输出类别数
        pretrained: 是否使用预训练权重（不适用于 CIFAR-10）

    返回:
        ResNet18ForCIFAR10 模型
    """
    model = ResNet18ForCIFAR10(num_classes=num_classes)
    return model


# ============================================================
# ResNet-50 支持
# ============================================================

class ResNet50ForCIFAR(_ResNetWrapper):
    """
    适用于 CIFAR 的 ResNet50（32x32 图像）。

    支持 CIFAR-10 (10 类) 和 CIFAR-100 (100 类)。
    """

    def __init__(self, num_classes: int = 100):
        """
        初始化适用于 CIFAR 的 ResNet50。

        参数:
            num_classes: 输出类别数（CIFAR-10=10, CIFAR-100=100）
        """
        super().__init__()

        from torchvision.models import resnet50

        # 加载标准 ResNet50 架构
        self.resnet = resnet50(weights=None)

        # 修改第一个卷积层以适应 32x32 输入
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 移除最大池化
        self.resnet.maxpool = nn.Identity()

        # 修改最终全连接层
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

class ResNet50ForImageNet(_ResNetWrapper):
    """
    标准 ResNet50 用于 ImageNet（224x224 图像）。
    """

    def __init__(self, num_classes: int = 1000, pretrained: bool = False):
        """
        初始化 ResNet50。

        参数:
            num_classes: 输出类别数（ImageNet=1000, Tiny-ImageNet=200）
            pretrained: 是否使用预训练权重
        """
        super().__init__()

        from torchvision.models import resnet50, ResNet50_Weights

        if pretrained:
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = resnet50(weights=None)

        # 如果类别数不是 1000，修改最终全连接层
        if num_classes != 1000:
            in_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

# ============================================================
# 工厂函数
# ============================================================

def get_resnet50_cifar100(pretrained: bool = False) -> ResNet50ForCIFAR:
    """创建 ResNet50 for CIFAR-100。"""
    return ResNet50ForCIFAR(num_classes=100)


def get_resnet50_tiny_imagenet(pretrained: bool = False) -> ResNet50ForImageNet:
    """创建 ResNet50 for Tiny-ImageNet (200 类)。"""
    return ResNet50ForImageNet(num_classes=200, pretrained=pretrained)


def get_resnet50_imagenet(pretrained: bool = False) -> ResNet50ForImageNet:
    """创建 ResNet50 for ImageNet (1000 类)。"""
    return ResNet50ForImageNet(num_classes=1000, pretrained=pretrained)


# ============================================================
# 通用工厂函数（用于 __init__.py 导出）
# ============================================================

def get_resnet18(num_classes: int = 10, pretrained: bool = False) -> ResNet18ForCIFAR10:
    """
    创建 ResNet18 模型的通用工厂函数。

    参数:
        num_classes: 输出类别数（默认 10 for CIFAR-10）
        pretrained: 是否使用预训练权重（CIFAR 不支持）

    返回:
        ResNet18ForCIFAR10 模型
    """
    return ResNet18ForCIFAR10(num_classes=num_classes)


def get_resnet50(num_classes: int = 1000, pretrained: bool = False, dataset: str = 'imagenet') -> nn.Module:
    """
    创建 ResNet50 模型的通用工厂函数。

    参数:
        num_classes: 输出类别数
        pretrained: 是否使用预训练权重
        dataset: 数据集类型 ('imagenet', 'cifar100', 'tiny-imagenet')

    返回:
        ResNet50 模型
    """
    if dataset == 'cifar100':
        return ResNet50ForCIFAR(num_classes=num_classes)
    elif dataset == 'tiny-imagenet':
        return ResNet50ForImageNet(num_classes=num_classes, pretrained=pretrained)
    else:  # imagenet
        return ResNet50ForImageNet(num_classes=num_classes, pretrained=pretrained)
