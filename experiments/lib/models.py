"""模型加载统一接口。

支持的模型:
  - gpt2-small, gpt2-medium
  - bert-base, bert-large
  - resnet18, resnet50
  - pythia-410m
  - vit-l-32, vit-b-16
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from dacp.utils.paths import resolve_model_source

# model_name -> model_family 映射
_FAMILY_MAP = {
    "gpt2-small": "gpt2",
    "gpt2-medium": "gpt2",
    "bert-base": "bert",
    "bert-large": "bert",
    "resnet18": "resnet",
    "resnet50": "resnet",
    "pythia-410m": "pythia",
    "pythia-1b": "pythia",
    "gpt2-large": "gpt2",
    "vit-l-32": "vit",
    "vit-b-16": "vit",
}


def get_model_type(model_name: str) -> str:
    """返回模型族名称（用于 param_schema 等）。"""
    family = _FAMILY_MAP.get(model_name)
    if family is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_FAMILY_MAP.keys())}")
    return family


def load_model(
    model_name: str,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    dataset_name: Optional[str] = None,
) -> Tuple[nn.Module, str]:
    """加载模型。

    参数:
        model_name: 模型名称
        pretrained: 是否加载预训练权重
        checkpoint_path: 微调检查点路径（可选）
        device: 设备
        dataset_name: 数据集名称（用于分类模型设置 num_classes 等）

    返回:
        (model, model_family) 元组
    """
    family = get_model_type(model_name)
    # 有检查点时无需下载预训练权重——从 config 创建结构，再加载 checkpoint
    effective_pretrained = pretrained if checkpoint_path is None else False
    model = _create_model(model_name, effective_pretrained, dataset_name=dataset_name)

    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    model = model.to(device)
    return model, family


def _create_model(
    model_name: str, pretrained: bool, dataset_name: Optional[str] = None
) -> nn.Module:
    """根据名称创建模型实例。"""
    if model_name in ("gpt2-small", "gpt2-medium"):
        return _create_gpt2(model_name, pretrained)
    if model_name in ("bert-base", "bert-large"):
        return _create_bert(model_name, pretrained, dataset_name)
    if model_name in ("resnet18", "resnet50"):
        return _create_resnet(model_name, pretrained)
    if model_name == "pythia-410m":
        return _create_pythia(pretrained)
    if model_name == "gpt2-large":
        return _create_gpt2_large(pretrained)
    if model_name == "pythia-1b":
        return _create_pythia_1b(pretrained)
    if model_name in ("vit-l-32", "vit-b-16"):
        return _create_vit(model_name, pretrained)
    raise ValueError(f"Unknown model: {model_name}")


def _create_gpt2(model_name: str, pretrained: bool) -> nn.Module:
    """Create one of the locally wrapped GPT-2 model variants."""
    from dacp.models.gpt2 import get_gpt2_medium, get_gpt2_small

    if model_name == "gpt2-small":
        return get_gpt2_small(pretrained=pretrained)
    if model_name == "gpt2-medium":
        return get_gpt2_medium(pretrained=pretrained)
    raise ValueError(f"Unsupported GPT-2 variant: {model_name}")


def _create_bert(
    model_name: str,
    pretrained: bool,
    dataset_name: Optional[str],
) -> nn.Module:
    """Create a BERT encoder or its GLUE classification head."""
    glue_num_labels = {"sst2": 2, "mnli": 3, "stsb": 1}
    if dataset_name not in glue_num_labels:
        from dacp.models.bert import get_bert_base, get_bert_large

        factory = get_bert_base if model_name == "bert-base" else get_bert_large
        return factory(pretrained=pretrained)

    from transformers import BertConfig, BertForSequenceClassification

    num_labels = glue_num_labels[dataset_name]
    if pretrained:
        hf_name = "bert-base-uncased" if model_name == "bert-base" else "bert-large-uncased"
        model = BertForSequenceClassification.from_pretrained(
            hf_name,
            num_labels=num_labels,
        )
    else:
        # Offline mode: create a matching config without a network request.
        bert_configs = {
            "bert-base": dict(
                vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            ),
            "bert-large": dict(
                vocab_size=30522,
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16,
                intermediate_size=4096,
            ),
        }
        model = BertForSequenceClassification(
            BertConfig(**bert_configs[model_name], num_labels=num_labels)
        )
    if dataset_name == "stsb":
        model.config.problem_type = "regression"
    return model


def _create_resnet(model_name: str, pretrained: bool) -> nn.Module:
    """Create one of the locally wrapped ResNet variants."""
    from dacp.models.resnet import get_resnet18, get_resnet50

    if model_name == "resnet18":
        return get_resnet18(pretrained=pretrained)
    if model_name == "resnet50":
        return get_resnet50(pretrained=pretrained)
    raise ValueError(f"Unsupported ResNet variant: {model_name}")


def _create_gpt2_large(pretrained: bool) -> nn.Module:
    """Create GPT-2 Large, preferring the repository's offline snapshot."""
    from transformers import GPT2Config, GPT2LMHeadModel

    source = resolve_model_source("gpt2-large", "gpt2-large")
    if pretrained:
        return GPT2LMHeadModel.from_pretrained(source)
    if Path(source).is_dir():
        config = GPT2Config.from_pretrained(source)
    else:
        config = GPT2Config(n_embd=1280, n_layer=36, n_head=20)
    return GPT2LMHeadModel(config)


def _create_pythia_1b(pretrained: bool) -> nn.Module:
    """Create Pythia-1B from a local config or the reviewed offline fallback."""
    return _create_pythia_variant("pythia-1b", pretrained)


def _create_pythia(pretrained: bool) -> nn.Module:
    """Create Pythia-410M from a local config or the reviewed offline fallback."""
    return _create_pythia_variant("pythia-410m", pretrained)


@dataclass(frozen=True)
class _PythiaSpec:
    remote_name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int


_PYTHIA_SPECS = {
    "pythia-410m": _PythiaSpec("EleutherAI/pythia-410m", 1024, 24, 16, 4096),
    "pythia-1b": _PythiaSpec("EleutherAI/pythia-1b", 2048, 16, 8, 8192),
}


def _create_pythia_variant(model_name: str, pretrained: bool) -> nn.Module:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    try:
        spec = _PYTHIA_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Pythia variant: {model_name}") from exc
    source = resolve_model_source(model_name, spec.remote_name)
    if pretrained:
        return GPTNeoXForCausalLM.from_pretrained(source)
    if Path(source).is_dir():
        config = GPTNeoXConfig.from_pretrained(source, local_files_only=True)
    else:
        config = GPTNeoXConfig(
            hidden_size=spec.hidden_size,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            intermediate_size=spec.intermediate_size,
            vocab_size=50304,
            max_position_embeddings=2048,
        )
    return GPTNeoXForCausalLM(config)


def _create_vit(model_name: str, pretrained: bool) -> nn.Module:
    """创建 ViT 模型。"""
    from transformers import ViTForImageClassification

    hf_name = {
        "vit-l-32": "google/vit-large-patch32-384",
        "vit-b-16": "google/vit-base-patch16-224",
    }[model_name]
    if pretrained:
        local_name = {
            "vit-l-32": "vit-large-patch32-384",
            "vit-b-16": "vit-base-patch16-224",
        }[model_name]
        return ViTForImageClassification.from_pretrained(resolve_model_source(local_name, hf_name))
    else:
        from transformers import ViTConfig

        configs = {
            "vit-l-32": dict(
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16,
                intermediate_size=4096,
                image_size=384,
                patch_size=32,
                num_labels=1000,
            ),
            "vit-b-16": dict(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                image_size=224,
                patch_size=16,
                num_labels=1000,
            ),
        }
        config = ViTConfig(**configs[model_name])
        return ViTForImageClassification(config)


def _extract_checkpoint_state(checkpoint) -> Mapping[str, torch.Tensor]:
    """Extract a model state dict from supported checkpoint payloads."""
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint model state must be a mapping")
    invalid = [
        name
        for name, value in checkpoint.items()
        if not isinstance(name, str) or not isinstance(value, torch.Tensor)
    ]
    if invalid:
        raise TypeError(f"checkpoint model state must map string keys to tensors: {invalid}")
    return checkpoint


def _load_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = True):
    """Load tensor-only model weights, failing closed on key mismatches by default."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _extract_checkpoint_state(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            "  检查点键不匹配: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    print(f"  已加载检查点: {checkpoint_path}")
