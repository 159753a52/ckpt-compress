"""数据加载统一接口。

支持的数据集:
  - wikitext2, wikitext103  (LM)
  - sst2, mnli              (CLS)
  - cifar10, cifar100        (CV)
  - alpaca                   (LM/Instruction)
  - imagenet                 (CV)
"""

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from experiments.lib.losses import SUPPORTED_TASK_TYPES

# task_type 映射
_TASK_MAP = {
    "wikitext2": "lm",
    "wikitext103": "lm",
    "sst2": "cls",
    "mnli": "cls",
    "mrpc": "cls",
    "qqp": "cls",
    "qnli": "cls",
    "cola": "cls",
    "rte": "cls",
    "stsb": "reg",
    "cifar10": "cv",
    "cifar100": "cv",
    "imagenet": "cv",
    "alpaca": "lm",
}


def make_causal_lm_labels(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Clone token IDs into labels and exclude padded positions from LM loss."""
    return _masked_causal_lm_labels(input_ids, input_ids, attention_mask)


def _masked_causal_lm_labels(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Validate a causal-LM batch and return detached padding-safe labels."""
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.dtype != torch.long:
        raise TypeError(f"input_ids must have dtype torch.long, got {input_ids.dtype}")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if labels.dtype != torch.long:
        raise TypeError(f"labels must have dtype torch.long, got {labels.dtype}")
    if labels.shape != input_ids.shape:
        raise ValueError(
            "labels must have the same shape as input_ids: "
            f"{tuple(labels.shape)} != {tuple(input_ids.shape)}"
        )
    if labels.device != input_ids.device:
        raise ValueError("labels and input_ids must be on the same device")

    masked_labels = labels.clone()
    if attention_mask is None:
        return masked_labels
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor")
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            "attention_mask must have the same shape as input_ids: "
            f"{tuple(attention_mask.shape)} != {tuple(input_ids.shape)}"
        )
    if attention_mask.device != input_ids.device:
        raise ValueError("attention_mask and input_ids must be on the same device")
    if attention_mask.is_complex() or not bool(
        torch.all((attention_mask == 0) | (attention_mask == 1)).item()
    ):
        raise ValueError("attention_mask must contain only 0 and 1")

    masked_labels.masked_fill_(attention_mask == 0, -100)
    return masked_labels


def tokenize_alpaca_batch(
    examples: Mapping[str, Sequence[str]],
    tokenizer,
    seq_length: int,
) -> Dict[str, torch.Tensor]:
    """Format and tokenize an Alpaca batch under the shared LM-label contract."""
    required_fields = ("instruction", "input", "output")
    missing = [field for field in required_fields if field not in examples]
    if missing:
        raise ValueError(f"Alpaca batch is missing fields: {missing}")

    field_lengths = {field: len(examples[field]) for field in required_fields}
    if len(set(field_lengths.values())) != 1:
        raise ValueError(f"Alpaca batch fields have different lengths: {field_lengths}")

    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
    ):
        prompt = f"### Instruction:\n{instruction}\n"
        if input_text:
            prompt += f"### Input:\n{input_text}\n"
        prompt += f"### Response:\n{output}"
        texts.append(prompt)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=seq_length,
        return_tensors="pt",
    )
    if "input_ids" not in encodings or "attention_mask" not in encodings:
        raise ValueError("tokenizer must return input_ids and attention_mask")
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise TypeError("tokenizer input_ids and attention_mask must be torch tensors")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": make_causal_lm_labels(input_ids, attention_mask),
    }


def get_task_type(dataset_name: str) -> str:
    """返回数据集对应的任务类型。"""
    if dataset_name not in _TASK_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(_TASK_MAP.keys())}")
    return _TASK_MAP[dataset_name]


def get_data_loaders(
    model_name: str,
    dataset_name: str,
    batch_size: int = 8,
    seq_length: int = 512,
    num_workers: int = 0,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader, str]:
    """获取训练和验证 DataLoader。

    参数:
        model_name: 模型名称（用于选择 tokenizer 等）
        dataset_name: 数据集名称
        batch_size: 批次大小
        seq_length: 序列长度（NLP 数据集用）
        num_workers: DataLoader 工作进程数
        data_dir: 数据存储目录

    返回:
        (train_loader, val_loader, task_type) 元组
    """
    task_type = get_task_type(dataset_name)

    if dataset_name in ("wikitext2", "wikitext103"):
        train_loader, val_loader = _get_wikitext_loaders(
            dataset_name, batch_size, seq_length, num_workers, data_dir
        )
    elif dataset_name in ("sst2", "mnli", "mrpc", "qqp", "qnli", "cola", "rte", "stsb"):
        train_loader, val_loader = _get_glue_loaders(
            model_name, dataset_name, batch_size, seq_length, num_workers, data_dir
        )
    elif dataset_name in ("cifar10", "cifar100"):
        train_loader, val_loader = _get_cifar_loaders(
            dataset_name, batch_size, num_workers, data_dir
        )
    elif dataset_name == "alpaca":
        train_loader, val_loader = _get_alpaca_loaders(
            model_name, batch_size, seq_length, num_workers, data_dir
        )
    elif dataset_name == "imagenet":
        train_loader, val_loader = _get_imagenet_loaders(batch_size, num_workers, data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_loader, val_loader, task_type


def cache_batches(
    data_loader: DataLoader,
    num_batches: int,
    task_type: str,
) -> List[Dict[str, torch.Tensor]]:
    """缓存数据批次到 CPU 内存。

    参数:
        data_loader: DataLoader
        num_batches: 缓存批次数
        task_type: 任务类型 ('lm', 'cls', 'cv', 'reg')

    返回:
        缓存的批次列表，每个元素为字典
    """
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unknown task_type: {task_type}. " f"Available: {sorted(SUPPORTED_TASK_TYPES)}"
        )
    if isinstance(num_batches, bool) or not isinstance(num_batches, int) or num_batches < 1:
        raise ValueError(f"num_batches must be a positive integer, got {num_batches}")

    cached = []
    data_iter = iter(data_loader)

    for _ in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        if task_type in ("lm",):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels")
            if labels is None:
                labels = input_ids
            masked_labels = _masked_causal_lm_labels(input_ids, labels, attention_mask)
            entry = {
                "input_ids": input_ids.clone(),
                "labels": masked_labels,
            }
            if attention_mask is not None:
                entry["attention_mask"] = attention_mask.clone()
            cached.append(entry)
        elif task_type in ("cls", "reg"):
            entry = {
                "input_ids": batch["input_ids"].clone(),
                "labels": batch["labels"].clone(),
            }
            if "attention_mask" in batch:
                entry["attention_mask"] = batch["attention_mask"].clone()
            cached.append(entry)
        elif task_type == "cv":
            if isinstance(batch, (list, tuple)):
                cached.append(
                    {
                        "images": batch[0].clone(),
                        "labels": batch[1].clone(),
                    }
                )
            else:
                cached.append(
                    {
                        "images": (
                            batch["images"].clone()
                            if "images" in batch
                            else batch["pixel_values"].clone()
                        ),
                        "labels": batch["labels"].clone(),
                    }
                )

    return cached


# ---- 内部加载函数 ----


def _find_wikitext_local_path(data_dir: str, dataset_name: str) -> Optional[str]:
    """Find a local WikiText directory without changing HF fallback behavior."""
    if not data_dir:
        return None
    root = Path(data_dir).expanduser()
    candidates = (root / dataset_name, root)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
        if (candidate / "train.txt").is_file() and (candidate / "valid.txt").is_file():
            return str(candidate)
    return None


def _get_wikitext_loaders(dataset_name, batch_size, seq_length, num_workers, data_dir):
    from dacp.utils.data_loader import get_wikitext2_dataloader, get_wikitext103_dataloader

    loader_fn = (
        get_wikitext2_dataloader if dataset_name == "wikitext2" else get_wikitext103_dataloader
    )
    local_path = _find_wikitext_local_path(data_dir, dataset_name)
    train_loader = loader_fn(
        split="train",
        batch_size=batch_size,
        seq_length=seq_length,
        num_workers=num_workers,
        shuffle=True,
        local_path=local_path,
    )
    val_loader = loader_fn(
        split="validation",
        batch_size=batch_size,
        seq_length=seq_length,
        num_workers=num_workers,
        shuffle=False,
        local_path=local_path,
    )
    return train_loader, val_loader


def _get_glue_loaders(model_name, dataset_name, batch_size, seq_length, num_workers, data_dir):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from dacp.utils.paths import resolve_model_source

    if model_name == "bert-large":
        tokenizer_name = resolve_model_source("bert-large-uncased", "bert-large-uncased")
    elif "bert" in model_name:
        tokenizer_name = resolve_model_source("bert-base-uncased", "bert-base-uncased")
    else:
        tokenizer_name = resolve_model_source("gpt2-medium", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=data_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("glue", dataset_name, cache_dir=data_dir)
    # GLUE 各任务字段映射
    _GLUE_KEYS = {
        "sst2": ("sentence", None),
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "rte": ("sentence1", "sentence2"),
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "qnli": ("question", "sentence"),
        "stsb": ("sentence1", "sentence2"),
    }
    key_a, key_b = _GLUE_KEYS[dataset_name]

    def tokenize(examples):
        args = (examples[key_a],) if key_b is None else (examples[key_a], examples[key_b])
        return tokenizer(*args, truncation=True, padding="max_length", max_length=seq_length)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_split = "validation_matched" if dataset_name == "mnli" else "validation"
    val_loader = DataLoader(
        dataset[val_split], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def _get_cifar_loaders(dataset_name, batch_size, num_workers, data_dir):
    from dacp.utils.data_loader import get_cifar10_loaders, get_cifar100_loaders

    loader_fn = get_cifar10_loaders if dataset_name == "cifar10" else get_cifar100_loaders
    return loader_fn(batch_size=batch_size, data_dir=data_dir, num_workers=num_workers)


def _get_alpaca_loaders(model_name, batch_size, seq_length, num_workers, data_dir):
    """加载 Alpaca 52K 指令微调数据集。"""
    import os

    from datasets import Dataset, load_dataset
    from torch.utils.data import DataLoader as DL
    from torch.utils.data import random_split
    from transformers import AutoTokenizer

    from dacp.utils.paths import resolve_model_source

    tokenizer_sources = {
        "pythia-410m": ("pythia-410m", "EleutherAI/pythia-410m"),
        "pythia-1b": ("pythia-1b", "EleutherAI/pythia-1b"),
    }
    if model_name not in tokenizer_sources:
        raise ValueError(f"Alpaca supports only {sorted(tokenizer_sources)}, got {model_name!r}")
    local_name, remote_name = tokenizer_sources[model_name]
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_source(local_name, remote_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 优先从本地 JSON 加载（离线模式）
    local_path = os.environ.get("ALPACA_LOCAL_PATH")
    if local_path is None:
        candidates = (
            Path(data_dir) / "alpaca" / "alpaca_data.json",
            Path(data_dir) / "alpaca_data.json",
        )
        local_path = next((str(path) for path in candidates if path.is_file()), None)
    if local_path and os.path.isfile(local_path):
        import json

        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
        print(f"  Alpaca: loaded {len(dataset)} samples from local JSON")
    else:
        dataset = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=data_dir)

    def format_and_tokenize(examples):
        return tokenize_alpaca_batch(examples, tokenizer, seq_length)

    dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")

    n_train, n_val = _alpaca_split_sizes(len(dataset))
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DL(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(0),
    )
    val_loader = DL(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def _alpaca_split_sizes(total_samples: int) -> tuple[int, int]:
    """Return deterministic non-empty train/validation sizes for Alpaca data."""
    if isinstance(total_samples, bool) or not isinstance(total_samples, int) or total_samples < 2:
        raise ValueError("Alpaca data must contain at least two samples")
    validation = min(1000, max(1, total_samples // 10))
    return total_samples - validation, validation


def _get_imagenet_loaders(batch_size, num_workers, data_dir):
    """加载 ImageNet-1K（需本地数据或 HF datasets）。"""
    import os

    from torch.utils.data import DataLoader as DL
    from torchvision import datasets, transforms

    # 环境变量覆盖 data_dir（用于不同服务器上的 ImageNet 路径）
    data_dir = os.environ.get("IMAGENET_DATA_DIR", data_dir)

    # 图像分辨率：ViT-L/32 使用 384，其他使用 224
    image_size = int(os.environ.get("IMAGE_SIZE", 224))
    resize_size = int(image_size * 256 / 224)  # 等比缩放

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dir = os.path.join(data_dir, "imagenet", "train")
    val_dir = os.path.join(data_dir, "imagenet", "val")

    if os.path.isdir(train_dir):
        train_ds = datasets.ImageFolder(train_dir, train_transform)
        val_ds = datasets.ImageFolder(val_dir, val_transform)
    else:
        # Fallback: HuggingFace datasets
        from datasets import load_dataset as hf_load

        raw = hf_load("imagenet-1k")

        class HFImageNet(torch.utils.data.Dataset):
            def __init__(self, hf_split, transform):
                self.data = hf_split
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                img = item["image"].convert("RGB")
                return self.transform(img), item["label"]

        train_ds = HFImageNet(raw["train"], train_transform)
        val_ds = HFImageNet(raw["validation"], val_transform)

    train_loader = DL(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DL(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
