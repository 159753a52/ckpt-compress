"""
BERT 模型工具。

用于 AdamPrune 实验，分析 BERT 架构的参数重要性分布。
支持 BERT-Base 和 BERT-Large 模型。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

try:
    from transformers import BertForMaskedLM, BertConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class BERTForExperiment(nn.Module):
    """
    用于实验的 BERT 模型封装。

    封装 HuggingFace 的 BertForMaskedLM，提供简化的接口。
    支持 BERT-Base 和 BERT-Large 两种配置。
    """

    def __init__(
        self,
        model_size: str = "base",
        pretrained: bool = False,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        初始化 BERT 模型。

        参数:
            model_size: 模型大小，'base' 或 'large'
            pretrained: 是否加载预训练权重
            local_files_only: 是否仅使用本地文件（不连接网络）
            cache_dir: 模型缓存目录
        """
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers 库未安装。请运行: pip install transformers"
            )

        self.model_size = model_size.lower()

        if self.model_size not in ["base", "large"]:
            raise ValueError(f"model_size 必须是 'base' 或 'large'，得到: {model_size}")

        if pretrained:
            # 加载预训练模型
            model_name = f"bert-{self.model_size}-uncased"
            kwargs = {'local_files_only': local_files_only}
            if cache_dir is not None:
                kwargs['cache_dir'] = cache_dir
            self.model = BertForMaskedLM.from_pretrained(
                model_name,
                **kwargs
            )
        else:
            # 随机初始化
            if self.model_size == "base":
                config = BertConfig(
                    vocab_size=30522,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    max_position_embeddings=512,
                )
            else:  # large
                config = BertConfig(
                    vocab_size=30522,
                    hidden_size=1024,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    intermediate_size=4096,
                    max_position_embeddings=512,
                )
            self.model = BertForMaskedLM(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        前向传播。

        参数:
            input_ids: 形状为 (batch_size, seq_len) 的输入 token ids
            attention_mask: 可选的注意力掩码
            token_type_ids: 可选的 token 类型 ids
            labels: 可选的标签（用于计算 MLM 损失）

        返回:
            模型输出（包含 logits 和可选的 loss）
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

    def parameters(self, recurse: bool = True):
        """返回模型参数。"""
        return self.model.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """返回命名的模型参数。"""
        return self.model.named_parameters(prefix, recurse)

    def state_dict(self, *args, **kwargs):
        """返回模型状态字典。"""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        """加载模型状态字典。"""
        return self.model.load_state_dict(state_dict, strict)

    def train(self, mode: bool = True):
        """设置训练模式。"""
        self.model.train(mode)
        return self

    def eval(self):
        """设置评估模式。"""
        self.model.eval()
        return self

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息。

        返回:
            包含模型配置信息的字典
        """
        config = self.model.config
        return {
            "model_size": self.model_size,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "max_position_embeddings": config.max_position_embeddings,
        }


def get_bert_base(
    pretrained: bool = False,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None
) -> BERTForExperiment:
    """
    创建 BERT-Base 模型的工厂函数。

    BERT-Base 配置:
    - 参数量: ~110M
    - 层数: 12
    - 隐藏维度: 768
    - 注意力头数: 12

    参数:
        pretrained: 是否加载预训练权重
        local_files_only: 是否仅使用本地文件（不连接网络）
        cache_dir: 模型缓存目录

    返回:
        BERTForExperiment 模型
    """
    return BERTForExperiment(
        model_size="base",
        pretrained=pretrained,
        local_files_only=local_files_only,
        cache_dir=cache_dir
    )


def get_bert_large(
    pretrained: bool = False,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None
) -> BERTForExperiment:
    """
    创建 BERT-Large 模型的工厂函数。

    BERT-Large 配置:
    - 参数量: ~340M
    - 层数: 24
    - 隐藏维度: 1024
    - 注意力头数: 16

    参数:
        pretrained: 是否加载预训练权重
        local_files_only: 是否仅使用本地文件（不连接网络）
        cache_dir: 模型缓存目录

    返回:
        BERTForExperiment 模型
    """
    return BERTForExperiment(
        model_size="large",
        pretrained=pretrained,
        local_files_only=local_files_only,
        cache_dir=cache_dir
    )
