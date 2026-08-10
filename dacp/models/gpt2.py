"""
GPT-2 模型工具。

用于 AdamPrune 实验，分析 Transformer 架构的参数重要性分布。
"""

import torch
from typing import Optional

from ._wrapper import DelegatingModel

try:
    from transformers import GPT2LMHeadModel, GPT2Config
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class GPT2ForExperiment(DelegatingModel):
    """
    用于实验的 GPT-2 模型封装。

    封装 HuggingFace 的 GPT2LMHeadModel，提供简化的接口。
    """

    def __init__(self, pretrained: bool = False):
        """
        初始化 GPT-2 模型。

        参数:
            pretrained: 是否加载预训练权重
        """
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers 库未安装。请运行: pip install transformers"
            )

        if pretrained:
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            # 使用 GPT-2 small 配置但随机初始化
            config = GPT2Config(
                vocab_size=50257,
                n_positions=1024,
                n_embd=768,
                n_layer=12,
                n_head=12,
            )
            self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            input_ids: 形状为 (batch_size, seq_len) 的输入 token ids

        返回:
            形状为 (batch_size, seq_len, vocab_size) 的 logits
        """
        outputs = self.model(input_ids)
        return outputs.logits

def get_gpt2_small(pretrained: bool = False) -> GPT2ForExperiment:
    """
    创建 GPT-2 small 模型的工厂函数。

    参数:
        pretrained: 是否加载预训练权重

    返回:
        GPT2ForExperiment 模型
    """
    return GPT2ForExperiment(pretrained=pretrained)


class GPT2MediumForExperiment(DelegatingModel):
    """
    用于实验的 GPT-2 Medium 模型封装。

    GPT-2 Medium 配置:
    - 参数量: ~355M
    - 层数: 24
    - 隐藏维度: 1024
    - 注意力头数: 16
    """

    def __init__(self, pretrained: bool = False, local_files_only: bool = False):
        """
        初始化 GPT-2 Medium 模型。

        参数:
            pretrained: 是否加载预训练权重
            local_files_only: 是否仅使用本地文件（不连接网络）
        """
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers 库未安装。请运行: pip install transformers"
            )

        if pretrained:
            self.model = GPT2LMHeadModel.from_pretrained(
                'gpt2-medium',
                local_files_only=local_files_only
            )
        else:
            # 使用 GPT-2 medium 配置但随机初始化
            config = GPT2Config(
                vocab_size=50257,
                n_positions=1024,
                n_embd=1024,
                n_layer=24,
                n_head=16,
            )
            self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        """
        前向传播。

        参数:
            input_ids: 形状为 (batch_size, seq_len) 的输入 token ids
            attention_mask: 可选的注意力掩码
            labels: 可选的标签（用于计算损失）

        返回:
            模型输出（包含 logits 和可选的 loss）
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

def get_gpt2_medium(pretrained: bool = False, local_files_only: bool = False) -> GPT2MediumForExperiment:
    """
    创建 GPT-2 medium 模型的工厂函数。

    参数:
        pretrained: 是否加载预训练权重
        local_files_only: 是否仅使用本地文件（不连接网络）

    返回:
        GPT2MediumForExperiment 模型
    """
    return GPT2MediumForExperiment(pretrained=pretrained, local_files_only=local_files_only)
