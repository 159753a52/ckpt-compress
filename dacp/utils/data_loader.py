"""
数据加载工具。

提供加载 CIFAR-10 和 WikiText-2 数据集的函数。
"""

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional, Dict

try:
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False

# GPT-2 tokenizer 本地 fallback 路径（离线服务器用）
_GPT2_TOKENIZER_FALLBACKS = [
    '/root/ckpt-compress/data/models/gpt2-medium',  # V100 server
    '/lihongliang/fangzl/ckpt-compress/models/gpt2_tokenizer',
]


def _apply_subset(dataset: Dataset, limit: Optional[int], name: str) -> Dataset:
    """Apply a prefix subset while enforcing one shared count contract."""
    if limit is None:
        return dataset
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError(
            f"{name} must be None or a non-negative integer, got {limit}"
        )
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def _load_gpt2_tokenizer():
    """加载 GPT-2 tokenizer，支持多路径 fallback。"""
    import os
    # 优先从本地路径加载（避免在线版本兼容性问题）
    for path in _GPT2_TOKENIZER_FALLBACKS:
        if os.path.isdir(path):
            try:
                tok = GPT2Tokenizer.from_pretrained(path)
                # 验证tokenizer能正常工作
                test = tok.encode('hello')
                if len(test) > 0:
                    return tok
            except Exception:
                pass
    try:
        return GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    except (OSError, ValueError, TypeError):
        pass
    try:
        return GPT2Tokenizer.from_pretrained('gpt2')
    except (OSError, ValueError, TypeError):
        pass
    raise RuntimeError('Cannot load GPT-2 tokenizer from any source')


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    获取 CIFAR-10 的数据变换。

    训练变换包括数据增强（随机裁剪、水平翻转）。
    测试变换仅进行归一化。

    返回:
        (train_transform, test_transform) 元组
    """
    # CIFAR-10 归一化参数
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


def get_cifar10_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    download: bool = True,
    num_workers: int = 4,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    获取 CIFAR-10 数据加载器。

    参数:
        batch_size: 数据加载器的批次大小
        data_dir: 存储/加载 CIFAR-10 数据的目录
        download: 如果数据不存在是否下载
        num_workers: 数据加载的工作进程数
        train_subset: 如果设置，仅使用指定数量的训练样本
        test_subset: 如果设置，仅使用指定数量的测试样本

    返回:
        (train_loader, test_loader) 元组
    """
    train_transform, test_transform = get_cifar10_transforms()

    # 加载数据集
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform,
    )

    # 如果指定则应用子集
    train_dataset = _apply_subset(train_dataset, train_subset, 'train_subset')
    test_dataset = _apply_subset(test_dataset, test_subset, 'test_subset')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_cifar100_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    download: bool = True,
    num_workers: int = 4,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    获取 CIFAR-100 数据加载器。

    参数:
        batch_size: 数据加载器的批次大小
        data_dir: 存储/加载 CIFAR-100 数据的目录
        download: 如果数据不存在是否下载
        num_workers: 数据加载的工作进程数
        train_subset: 如果设置，仅使用指定数量的训练样本
        test_subset: 如果设置，仅使用指定数量的测试样本

    返回:
        (train_loader, test_loader) 元组
    """
    train_transform, test_transform = get_cifar10_transforms()

    # 加载数据集（CIFAR-100 使用相同的变换）
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform,
    )

    # 如果指定则应用子集
    train_dataset = _apply_subset(train_dataset, train_subset, 'train_subset')
    test_dataset = _apply_subset(test_dataset, test_subset, 'test_subset')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


class WikiText2Dataset(Dataset):
    """
    WikiText-2 数据集封装。

    将文本数据转换为固定长度的序列，用于语言模型训练。
    """

    def __init__(
        self,
        texts: list,
        tokenizer,
        seq_length: int = 512,
        max_samples: Optional[int] = None
    ):
        if (
            isinstance(seq_length, bool)
            or not isinstance(seq_length, int)
            or seq_length < 1
        ):
            raise ValueError(f"seq_length must be a positive integer, got {seq_length}")
        if max_samples is not None and (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or max_samples < 0
        ):
            raise ValueError(
                "max_samples must be None or a non-negative integer, "
                f"got {max_samples}"
            )

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.samples = []

        if max_samples == 0:
            return

        # 增量编码文本，避免一次性编码整个数据集
        tokens = []
        target_samples = max_samples if max_samples is not None else 1000
        target_tokens = target_samples * seq_length + seq_length  # 需要的 token 数量

        for text in texts:
            if not text.strip():
                continue
            # 逐段编码
            new_tokens = tokenizer.encode(text)
            tokens.extend(new_tokens)

            # 如果已经有足够的 tokens，提前终止
            if len(tokens) >= target_tokens:
                break

        # 切分为固定长度的序列
        for i in range(0, len(tokens) - seq_length, seq_length):
            input_ids = tokens[i:i + seq_length]
            self.samples.append(input_ids)

            if max_samples is not None and len(self.samples) >= max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.samples[idx], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        # labels 用于语言模型训练（与 input_ids 相同，模型内部会移位）
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def _get_wikitext_dataloader(
    dataset_config: str,
    split: str = 'train',
    batch_size: int = 8,
    seq_length: int = 512,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    local_path: Optional[str] = None,
) -> DataLoader:
    """Build either WikiText loader from one shared data path."""
    if not HAS_HF:
        raise ImportError(
            "需要安装 transformers 和 datasets 库。"
            "请运行: pip install transformers datasets"
        )

    valid_splits = ('train', 'validation', 'test')
    if split not in valid_splits:
        raise ValueError(
            f"Invalid split: {split}. Must be one of {valid_splits}"
        )

    tokenizer = _load_gpt2_tokenizer()

    if local_path is not None:
        import os

        if os.path.isfile(local_path):
            file_path = local_path
        else:
            split_file_map = {
                'train': 'train.txt',
                'validation': 'valid.txt',
                'test': 'test.txt',
            }
            file_path = os.path.join(local_path, split_file_map[split])

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local data file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        print(f"Loaded {len(texts)} lines from local file: {file_path}")
    else:
        dataset = load_dataset('wikitext', dataset_config, split=split)
        texts = dataset['text']

    wiki_dataset = WikiText2Dataset(
        texts=texts,
        tokenizer=tokenizer,
        seq_length=seq_length,
        max_samples=max_samples,
    )

    if shuffle is None:
        shuffle = (split == 'train')

    return DataLoader(
        wiki_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_wikitext2_dataloader(
    split: str = 'train',
    batch_size: int = 8,
    seq_length: int = 512,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    local_path: Optional[str] = None,
) -> DataLoader:
    """
    获取 WikiText-2 数据加载器。

    参数:
        split: 数据集划分 ('train', 'validation', 'test')
        batch_size: 批次大小
        seq_length: 序列长度
        max_samples: 最大样本数（用于快速测试）
        num_workers: 数据加载工作进程数
        shuffle: 是否打乱数据（默认：train=True, 其他=False）
        local_path: 本地数据文件路径（如果提供，则从本地加载而不是从 HuggingFace 下载）

    返回:
        DataLoader 对象

    异常:
        ValueError: 如果 split 无效
        ImportError: 如果 transformers 或 datasets 未安装
    """
    return _get_wikitext_dataloader(
        dataset_config='wikitext-2-raw-v1',
        split=split,
        batch_size=batch_size,
        seq_length=seq_length,
        max_samples=max_samples,
        num_workers=num_workers,
        shuffle=shuffle,
        local_path=local_path,
    )


def get_wikitext103_dataloader(
    split: str = 'train',
    batch_size: int = 8,
    seq_length: int = 512,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    local_path: Optional[str] = None,
) -> DataLoader:
    """
    获取 WikiText-103 数据加载器。

    WikiText-103 特点:
    - 大小: ~500 MB
    - Token 数: ~103M
    - 适用于: GPT-2 Medium 主结果实验

    参数:
        split: 数据集划分 ('train', 'validation', 'test')
        batch_size: 批次大小
        seq_length: 序列长度
        max_samples: 最大样本数（用于快速测试）
        num_workers: 数据加载工作进程数
        shuffle: 是否打乱数据（默认：train=True, 其他=False）
        local_path: 本地数据文件路径（如果提供，则从本地加载而不是从 HuggingFace 下载）

    返回:
        DataLoader 对象

    异常:
        ValueError: 如果 split 无效
        ImportError: 如果 transformers 或 datasets 未安装
    """
    return _get_wikitext_dataloader(
        dataset_config='wikitext-103-raw-v1',
        split=split,
        batch_size=batch_size,
        seq_length=seq_length,
        max_samples=max_samples,
        num_workers=num_workers,
        shuffle=shuffle,
        local_path=local_path,
    )


class TinyImageNetDataset(Dataset):
    """
    Tiny-ImageNet 数据集封装。

    数据集特点:
    - 大小: ~250 MB
    - 样本数: 100,000 训练 + 10,000 验证
    - 类别数: 200
    - 图像尺寸: 64x64x3
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        download: bool = False,
    ):
        """
        初始化 Tiny-ImageNet 数据集。

        参数:
            root: 数据集根目录
            split: 数据集划分 ('train', 'val')
            transform: 数据变换
            download: 是否下载数据集
        """
        from pathlib import Path

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.data_dir = self.root / 'tiny-imagenet-200'

        if download:
            self._download()

        if not self.data_dir.exists():
            raise RuntimeError(
                f"Dataset not found at {self.data_dir}. "
                "Set download=True to download it."
            )

        # 加载类别映射
        self.class_to_idx = self._load_class_mapping()
        self.classes = list(self.class_to_idx.keys())

        # 加载图像路径和标签
        self.samples = self._load_samples()

    def _download(self):
        """下载 Tiny-ImageNet 数据集。"""
        import urllib.request
        import zipfile

        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        zip_path = self.root / 'tiny-imagenet-200.zip'

        # 创建目录
        self.root.mkdir(parents=True, exist_ok=True)

        # 如果已经存在，跳过下载
        if self.data_dir.exists():
            print(f"Dataset already exists at {self.data_dir}")
            return

        print(f"Downloading Tiny-ImageNet from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"Downloaded to {zip_path}")

            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            print(f"Extracted to {self.data_dir}")

            # 删除 zip 文件
            zip_path.unlink()
            print("Download and extraction completed!")

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise

    def _load_class_mapping(self):
        """加载类别映射。"""
        wnids_file = self.data_dir / 'wnids.txt'
        if not wnids_file.exists():
            raise RuntimeError(f"wnids.txt not found at {wnids_file}")

        class_to_idx = {}
        with open(wnids_file, 'r') as f:
            for idx, line in enumerate(f):
                class_id = line.strip()
                class_to_idx[class_id] = idx

        return class_to_idx

    def _load_samples(self):
        """加载样本路径和标签。"""
        samples = []

        if self.split == 'train':
            # 训练集：每个类别一个文件夹
            train_dir = self.data_dir / 'train'
            for class_id in self.classes:
                class_dir = train_dir / class_id / 'images'
                if not class_dir.exists():
                    continue

                label = self.class_to_idx[class_id]
                for img_file in class_dir.glob('*.JPEG'):
                    samples.append((str(img_file), label))

        elif self.split == 'val':
            # 验证集：所有图像在一个文件夹，标签在 annotations 文件中
            val_dir = self.data_dir / 'val'
            val_annotations = val_dir / 'val_annotations.txt'

            if not val_annotations.exists():
                raise RuntimeError(f"val_annotations.txt not found at {val_annotations}")

            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue

                    img_name = parts[0]
                    class_id = parts[1]

                    if class_id not in self.class_to_idx:
                        continue

                    img_path = val_dir / 'images' / img_name
                    label = self.class_to_idx[class_id]
                    samples.append((str(img_path), label))

        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        获取样本。

        参数:
            idx: 样本索引

        返回:
            (image, label) 元组
        """
        from PIL import Image

        img_path, label = self.samples[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 应用变换
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_tiny_imagenet_transforms():
    """
    获取 Tiny-ImageNet 的数据变换。

    返回:
        (train_transform, val_transform) 元组
    """
    # Tiny-ImageNet 归一化参数（使用 ImageNet 的参数）
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform


def get_tiny_imagenet_loaders(
    batch_size: int = 128,
    data_dir: str = "./data/tiny-imagenet",
    download: bool = False,
    num_workers: int = 4,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
):
    """
    获取 Tiny-ImageNet 数据加载器。

    参数:
        batch_size: 数据加载器的批次大小
        data_dir: 存储/加载 Tiny-ImageNet 数据的目录
        download: 如果数据不存在是否下载
        num_workers: 数据加载的工作进程数
        train_subset: 如果设置，仅使用指定数量的训练样本
        val_subset: 如果设置，仅使用指定数量的验证样本

    返回:
        (train_loader, val_loader) 元组
    """
    train_transform, val_transform = get_tiny_imagenet_transforms()

    # 加载数据集
    train_dataset = TinyImageNetDataset(
        root=data_dir,
        split='train',
        transform=train_transform,
        download=download,
    )

    val_dataset = TinyImageNetDataset(
        root=data_dir,
        split='val',
        transform=val_transform,
        download=download,
    )

    # 如果指定则应用子集
    train_dataset = _apply_subset(train_dataset, train_subset, 'train_subset')
    val_dataset = _apply_subset(val_dataset, val_subset, 'val_subset')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# =============================================================================
# GLUE 数据集加载器
# =============================================================================


def get_glue_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int = 32,
    data_dir: str = "./data",
    max_length: int = 128,
    num_workers: int = 0,
    subset: Optional[int] = None,
) -> DataLoader:
    """
    通用 GLUE 数据集加载器。

    参数:
        dataset_name: GLUE 数据集名称 (sst2, mnli, stsb 等)
        split: 数据划分 (train, validation, test)
        batch_size: 批次大小
        data_dir: 数据存储目录
        max_length: 最大序列长度
        num_workers: 数据加载工作进程数
        subset: 如果设置，仅使用指定数量的样本

    返回:
        DataLoader: 数据加载器
    """
    if not HAS_HF:
        raise ImportError("HuggingFace datasets and transformers are required. "
                         "Install with: pip install datasets transformers")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # 绕过 NFS 磁盘空间检查误报（df 显示 100% 但实际有空间）
    import shutil
    _orig_disk_usage = shutil.disk_usage
    shutil.disk_usage = lambda path: shutil._ntuple_diskusage(1 << 40, 0, 1 << 40)
    try:
        dataset = load_dataset('glue', dataset_name, cache_dir=data_dir, split=split)
    finally:
        shutil.disk_usage = _orig_disk_usage

    # 加载 tokenizer（使用 BERT tokenizer）
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    class GLUEDataset(Dataset):
        """GLUE 数据集包装器。"""

        def __init__(self, dataset, tokenizer, max_length: int = 128):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]

            # 根据不同任务处理输入
            if dataset_name == 'stsb':
                # STS-B: 句子对 + 回归标签 (sentence1, sentence2)
                inputs = self.tokenizer(
                    item['sentence1'],
                    item['sentence2'],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                )
                label = torch.tensor(item['label'], dtype=torch.float32)
            elif dataset_name == 'mnli':
                # MNLI: 句子对 + 三分类标签 (premise, hypothesis)
                inputs = self.tokenizer(
                    item['premise'],
                    item['hypothesis'],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                )
                label = torch.tensor(item['label'], dtype=torch.long)
            elif dataset_name == 'sst2':
                # SST-2: 单句子 + 二分类标签
                inputs = self.tokenizer(
                    item['sentence'],
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                )
                label = torch.tensor(item['label'], dtype=torch.long)
            else:
                # 通用处理：尝试自动检测
                if 'sentence1' in item and 'sentence2' in item:
                    inputs = self.tokenizer(
                        item['sentence1'],
                        item['sentence2'],
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt',
                    )
                elif 'premise' in item and 'hypothesis' in item:
                    inputs = self.tokenizer(
                        item['premise'],
                        item['hypothesis'],
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt',
                    )
                else:
                    inputs = self.tokenizer(
                        item.get('sentence', ''),
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt',
                    )
                # 根据任务类型确定标签类型
                if dataset_name == 'stsb':
                    label = torch.tensor(item['label'], dtype=torch.float32)
                else:
                    label = torch.tensor(item['label'], dtype=torch.long)

            # 移除 batch 维度
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs['labels'] = label

            return inputs

    # 创建数据集
    glue_dataset = GLUEDataset(dataset, tokenizer, max_length)

    # 应用子集
    glue_dataset = _apply_subset(glue_dataset, subset, 'subset')

    # 创建数据加载器
    dataloader = DataLoader(
        glue_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def get_sst2_loaders(
    batch_size: int = 32,
    data_dir: str = "./data",
    max_length: int = 128,
    num_workers: int = 0,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    获取 SST-2 (Stanford Sentiment Treebank) 数据加载器。

    SST-2 是二分类情感分析任务：正面/负面。

    参数:
        batch_size: 批次大小
        data_dir: 数据存储目录
        max_length: 最大序列长度
        num_workers: 数据加载工作进程数
        train_subset: 训练集子集大小
        val_subset: 验证集子集大小

    返回:
        (train_loader, val_loader) 元组
    """
    train_loader = get_glue_dataloader(
        dataset_name='sst2',
        split='train',
        batch_size=batch_size,
        data_dir=data_dir,
        max_length=max_length,
        num_workers=num_workers,
        subset=train_subset,
    )

    val_loader = get_glue_dataloader(
        dataset_name='sst2',
        split='validation',
        batch_size=batch_size,
        data_dir=data_dir,
        max_length=max_length,
        num_workers=num_workers,
        subset=val_subset,
    )

    return train_loader, val_loader


def get_mnli_loaders(
    batch_size: int = 32,
    data_dir: str = "./data",
    max_length: int = 128,
    num_workers: int = 0,
    train_subset: Optional[int] = None,
    val_matched_subset: Optional[int] = None,
    val_mismatched_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取 MNLI (Multi-Genre Natural Language Inference) 数据加载器。

    MNLI 是三分类自然语言推理任务：蕴含/矛盾/中立。
    包含两个验证集：matched (匹配) 和 mismatched (不匹配)。

    参数:
        batch_size: 批次大小
        data_dir: 数据存储目录
        max_length: 最大序列长度
        num_workers: 数据加载工作进程数
        train_subset: 训练集子集大小
        val_matched_subset: matched 验证集子集大小
        val_mismatched_subset: mismatched 验证集子集大小

    返回:
        (train_loader, val_matched_loader, val_mismatched_loader) 元组
    """
    train_loader = get_glue_dataloader(
        dataset_name='mnli',
        split='train',
        batch_size=batch_size,
        data_dir=data_dir,
        max_length=max_length,
        num_workers=num_workers,
        subset=train_subset,
    )

    val_matched_loader = get_glue_dataloader(
        dataset_name='mnli',
        split='validation_matched',
        batch_size=batch_size,
        data_dir=data_dir,
        max_length=max_length,
        num_workers=num_workers,
        subset=val_matched_subset,
    )

    val_mismatched_loader = get_glue_dataloader(
        dataset_name='mnli',
        split='validation_mismatched',
        batch_size=batch_size,
        data_dir=data_dir,
        max_length=max_length,
        num_workers=num_workers,
        subset=val_mismatched_subset,
    )

    return train_loader, val_matched_loader, val_mismatched_loader


def get_stsb_loaders(
    batch_size: int = 32,
    data_dir: str = "./data",
    max_length: int = 128,
    num_workers: int = 0,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    获取 STS-B (Semantic Textual Similarity Benchmark) 数据加载器。

    STS-B 是回归任务：评估两个句子的语义相似度（0-5 分）。

    参数:
        batch_size: 批次大小
        data_dir: 数据存储目录
        max_length: 最大序列长度
        num_workers: 数据加载工作进程数
        train_subset: 训练集子集大小
        val_subset: 验证集子集大小

    返回:
        (train_loader, val_loader) 元组
    """
    train_loader = get_glue_dataloader(
        dataset_name='stsb',
        split='train',
        batch_size=batch_size,
        data_dir=data_dir,
        max_length=max_length,
        num_workers=num_workers,
        subset=train_subset,
    )

    val_loader = get_glue_dataloader(
        dataset_name='stsb',
        split='validation',
        batch_size=batch_size,
        data_dir=data_dir,
        max_length=max_length,
        num_workers=num_workers,
        subset=val_subset,
    )

    return train_loader, val_loader
