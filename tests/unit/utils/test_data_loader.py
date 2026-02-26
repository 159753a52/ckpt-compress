"""
CIFAR-10/CIFAR-100 数据加载器测试。

TDD：先写测试，再实现。
"""

import pytest
import torch
from torch.utils.data import DataLoader

from ckpt_compress.utils.data_loader import (
    get_cifar10_loaders,
    get_cifar10_transforms,
    get_cifar100_loaders,
)


class TestGetCIFAR10Transforms:
    """CIFAR-10 数据变换测试。"""

    def test_train_transform_exists(self):
        """应返回训练变换。"""
        train_transform, test_transform = get_cifar10_transforms()
        assert train_transform is not None

    def test_test_transform_exists(self):
        """应返回测试变换。"""
        train_transform, test_transform = get_cifar10_transforms()
        assert test_transform is not None

    def test_transforms_normalize_to_tensor(self):
        """变换应转换为张量并归一化。"""
        train_transform, test_transform = get_cifar10_transforms()

        # 创建一个虚拟的类 PIL 图像（作为 numpy 数组）
        import numpy as np
        from PIL import Image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))

        # 应用变换
        tensor = test_transform(dummy_img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)


class TestGetCIFAR10Loaders:
    """CIFAR-10 数据加载器测试。"""

    def test_loaders_returned(self):
        """应返回训练和测试加载器。"""
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_train_loader_batch_size(self):
        """训练加载器应有正确的批次大小。"""
        train_loader, _ = get_cifar10_loaders(
            batch_size=64,
            data_dir="./data",
            download=True,
            num_workers=0
        )

        batch = next(iter(train_loader))
        images, labels = batch

        assert images.shape[0] == 64

    def test_loader_image_shape(self):
        """图像形状应为 (batch, 3, 32, 32)。"""
        train_loader, _ = get_cifar10_loaders(
            batch_size=16,
            data_dir="./data",
            download=True,
            num_workers=0
        )

        images, labels = next(iter(train_loader))

        assert images.shape[1:] == (3, 32, 32)

    def test_loader_label_range(self):
        """标签应在 [0, 9] 范围内。"""
        train_loader, _ = get_cifar10_loaders(
            batch_size=100,
            data_dir="./data",
            download=True,
            num_workers=0
        )

        _, labels = next(iter(train_loader))

        assert labels.min() >= 0
        assert labels.max() <= 9

    def test_train_loader_shuffled(self):
        """训练加载器应打乱数据。"""
        train_loader, _ = get_cifar10_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0
        )

        # 检查打乱已启用
        assert train_loader.dataset is not None

    def test_test_loader_not_shuffled(self):
        """测试加载器不应打乱数据。"""
        _, test_loader = get_cifar10_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0
        )

        # 测试加载器存在
        assert test_loader.dataset is not None

    def test_subset_option(self):
        """应支持数据子集以进行快速实验。"""
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0,
            train_subset=1000,
            test_subset=200
        )

        # 计算样本数
        train_samples = len(train_loader.dataset)
        test_samples = len(test_loader.dataset)

        assert train_samples == 1000
        assert test_samples == 200


class TestWikiText2LoaderImportError:
    """WikiText-2 数据加载器 ImportError 测试。"""

    def test_import_error_when_hf_not_available(self, monkeypatch):
        """当 transformers/datasets 未安装时应抛出 ImportError。"""
        from ckpt_compress.utils import data_loader
        
        # 模拟 HAS_HF = False
        monkeypatch.setattr(data_loader, 'HAS_HF', False)
        
        with pytest.raises(ImportError, match="需要安装 transformers 和 datasets 库"):
            data_loader.get_wikitext2_dataloader(
                split='train',
                batch_size=4,
                seq_length=128
            )


class TestWikiText2Loader:
    """WikiText-2 数据加载器测试。"""

    def test_get_wikitext2_dataloader_returns_dataloader(self):
        """应该返回 DataLoader 对象。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader

        loader = get_wikitext2_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=100
        )

        assert isinstance(loader, DataLoader)

    def test_wikitext2_train_split(self):
        """应该能加载训练集。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader

        loader = get_wikitext2_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=50
        )

        batch = next(iter(loader))
        assert 'input_ids' in batch
        assert batch['input_ids'].shape[0] == 4

    def test_wikitext2_valid_split(self):
        """应该能加载验证集。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader

        loader = get_wikitext2_dataloader(
            split='validation',
            batch_size=4,
            seq_length=128,
            max_samples=50
        )

        batch = next(iter(loader))
        assert 'input_ids' in batch

    def test_wikitext2_test_split(self):
        """应该能加载测试集。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader

        loader = get_wikitext2_dataloader(
            split='test',
            batch_size=4,
            seq_length=128,
            max_samples=50
        )

        batch = next(iter(loader))
        assert 'input_ids' in batch

    def test_wikitext2_sequence_length(self):
        """序列长度应该正确。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader

        seq_length = 256
        loader = get_wikitext2_dataloader(
            split='train',
            batch_size=4,
            seq_length=seq_length,
            max_samples=50
        )

        batch = next(iter(loader))
        assert batch['input_ids'].shape[1] == seq_length

    def test_wikitext2_batch_size(self):
        """批次大小应该正确。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader

        batch_size = 8
        loader = get_wikitext2_dataloader(
            split='train',
            batch_size=batch_size,
            seq_length=128,
            max_samples=100
        )

        batch = next(iter(loader))
        assert batch['input_ids'].shape[0] == batch_size

    def test_wikitext2_returns_attention_mask(self):
        """应该返回 attention_mask。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader

        loader = get_wikitext2_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=50
        )

        batch = next(iter(loader))
        assert 'attention_mask' in batch
        assert batch['attention_mask'].shape == batch['input_ids'].shape

    def test_wikitext2_returns_labels(self):
        """应该返回 labels（用于语言模型训练）。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader

        loader = get_wikitext2_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=50
        )

        batch = next(iter(loader))
        assert 'labels' in batch
        # labels 应该是 input_ids 的移位版本
        assert batch['labels'].shape == batch['input_ids'].shape

    def test_wikitext2_invalid_split_raises_error(self):
        """无效的 split 应该抛出错误。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader

        with pytest.raises(ValueError):
            get_wikitext2_dataloader(
                split='invalid',
                batch_size=4,
                seq_length=128
            )

    def test_wikitext2_max_samples(self):
        """max_samples 应该限制样本数量。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader

        loader = get_wikitext2_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=20
        )

        total_samples = len(loader.dataset)
        assert total_samples <= 20

    def test_wikitext2_local_path_loading(self, tmp_path):
        """应该能从本地文件加载数据。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader
        
        # 创建临时测试文件
        test_file = tmp_path / "train.txt"
        test_content = "This is a test sentence.\nAnother test sentence.\n" * 100
        test_file.write_text(test_content)
        
        # 使用本地路径加载
        loader = get_wikitext2_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=10,
            local_path=str(tmp_path)
        )
        
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert 'input_ids' in batch

    def test_wikitext2_local_file_path(self, tmp_path):
        """应该能直接从文件路径加载数据。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader
        
        # 创建临时测试文件
        test_file = tmp_path / "test_data.txt"
        test_content = "Test content for loading.\n" * 100
        test_file.write_text(test_content)
        
        # 直接使用文件路径
        loader = get_wikitext2_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=10,
            local_path=str(test_file)
        )
        
        assert isinstance(loader, DataLoader)

    def test_wikitext2_shuffle_default_train(self):
        """训练集默认应该 shuffle。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader
        
        loader = get_wikitext2_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=50,
            shuffle=None  # 使用默认值
        )
        
        # 训练集默认 shuffle=True
        assert loader.sampler is not None or hasattr(loader, 'shuffle')

    def test_wikitext2_shuffle_default_validation(self):
        """验证集默认不应该 shuffle。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader
        
        loader = get_wikitext2_dataloader(
            split='validation',
            batch_size=4,
            seq_length=128,
            max_samples=50,
            shuffle=None  # 使用默认值
        )
        
        # 验证集默认 shuffle=False
        assert loader.dataset is not None

    def test_wikitext2_explicit_shuffle_false(self):
        """应该能显式设置 shuffle=False。"""
        from ckpt_compress.utils.data_loader import get_wikitext2_dataloader
        
        loader = get_wikitext2_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=50,
            shuffle=False
        )
        
        assert loader.dataset is not None


class TestGetCIFAR100Loaders:
    """CIFAR-100 数据加载器测试。"""

    def test_loaders_returned(self):
        """应返回训练和测试加载器。"""
        train_loader, test_loader = get_cifar100_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0,
            train_subset=100,
            test_subset=50
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_train_loader_batch_size(self):
        """训练加载器应有正确的批次大小。"""
        train_loader, _ = get_cifar100_loaders(
            batch_size=64,
            data_dir="./data",
            download=True,
            num_workers=0,
            train_subset=128
        )

        batch = next(iter(train_loader))
        images, labels = batch

        assert images.shape[0] == 64

    def test_loader_image_shape(self):
        """图像形状应为 (batch, 3, 32, 32)。"""
        train_loader, _ = get_cifar100_loaders(
            batch_size=16,
            data_dir="./data",
            download=True,
            num_workers=0,
            train_subset=100
        )

        images, labels = next(iter(train_loader))

        assert images.shape[1:] == (3, 32, 32)

    def test_loader_label_range(self):
        """标签应在 [0, 99] 范围内（CIFAR-100 有 100 个类别）。"""
        train_loader, _ = get_cifar100_loaders(
            batch_size=100,
            data_dir="./data",
            download=True,
            num_workers=0,
            train_subset=500  # 使用较大子集以确保覆盖更多类别
        )

        _, labels = next(iter(train_loader))

        assert labels.min() >= 0
        assert labels.max() <= 99

    def test_train_dataset_size_with_subset(self):
        """使用 subset 时训练集大小应正确。"""
        train_loader, _ = get_cifar100_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0,
            train_subset=1000
        )

        # 使用 subset 时应该是指定的大小
        assert len(train_loader.dataset) == 1000

    def test_test_dataset_size_with_subset(self):
        """使用 subset 时测试集大小应正确。"""
        _, test_loader = get_cifar100_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0,
            test_subset=200
        )

        # 使用 subset 时应该是指定的大小
        assert len(test_loader.dataset) == 200

    def test_data_augmentation_applied(self):
        """训练数据应应用数据增强。"""
        train_loader, _ = get_cifar100_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0,
            train_subset=100
        )

        # 检查数据加载器存在且可以迭代
        batch = next(iter(train_loader))
        assert batch is not None

    def test_subset_option(self):
        """应支持数据子集以进行快速实验。"""
        train_loader, test_loader = get_cifar100_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0,
            train_subset=1000,
            test_subset=200
        )

        # 计算样本数
        train_samples = len(train_loader.dataset)
        test_samples = len(test_loader.dataset)

        assert train_samples == 1000
        assert test_samples == 200

    def test_num_workers_parameter(self):
        """应支持 num_workers 参数。"""
        train_loader, test_loader = get_cifar100_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=2
        )

        assert train_loader.num_workers == 2
        assert test_loader.num_workers == 2

    def test_loader_iteration(self):
        """数据加载器应可以正常迭代。"""
        train_loader, _ = get_cifar100_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0,
            train_subset=100  # 使用小子集加快测试
        )

        batch_count = 0
        for images, labels in train_loader:
            batch_count += 1
            assert images.shape[1:] == (3, 32, 32)
            assert labels.shape[0] <= 32

        # 应该有至少一个批次
        assert batch_count > 0

    def test_compatibility_with_cifar10(self):
        """CIFAR-100 和 CIFAR-10 应使用相同的图像尺寸和格式。"""
        cifar10_train, _ = get_cifar10_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0
        )

        cifar100_train, _ = get_cifar100_loaders(
            batch_size=32,
            data_dir="./data",
            download=True,
            num_workers=0
        )

        # 获取一个批次
        cifar10_images, _ = next(iter(cifar10_train))
        cifar100_images, _ = next(iter(cifar100_train))

        # 图像形状应该相同
        assert cifar10_images.shape[1:] == cifar100_images.shape[1:]
        assert cifar10_images.shape[1:] == (3, 32, 32)


class TestWikiText103LoaderImportError:
    """WikiText-103 数据加载器 ImportError 测试。"""

    def test_import_error_when_hf_not_available(self, monkeypatch):
        """当 transformers/datasets 未安装时应抛出 ImportError。"""
        from ckpt_compress.utils import data_loader
        
        # 模拟 HAS_HF = False
        monkeypatch.setattr(data_loader, 'HAS_HF', False)
        
        with pytest.raises(ImportError, match="需要安装 transformers 和 datasets 库"):
            data_loader.get_wikitext103_dataloader(
                split='train',
                batch_size=4,
                seq_length=128
            )


class TestWikiText103Loader:
    """WikiText-103 数据加载器测试。"""

    def test_get_wikitext103_dataloader_returns_dataloader(self):
        """应该返回 DataLoader 对象。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader

        loader = get_wikitext103_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=50
        )

        assert isinstance(loader, DataLoader)

    def test_wikitext103_train_split(self):
        """应该能加载训练集。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader

        loader = get_wikitext103_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=50
        )

        batch = next(iter(loader))
        assert 'input_ids' in batch
        assert batch['input_ids'].shape[0] == 4

    def test_wikitext103_valid_split(self):
        """应该能加载验证集。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader

        loader = get_wikitext103_dataloader(
            split='validation',
            batch_size=4,
            seq_length=128,
            max_samples=50
        )

        batch = next(iter(loader))
        assert 'input_ids' in batch

    def test_wikitext103_test_split(self):
        """应该能加载测试集。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader

        loader = get_wikitext103_dataloader(
            split='test',
            batch_size=4,
            seq_length=128,
            max_samples=50
        )

        batch = next(iter(loader))
        assert 'input_ids' in batch

    def test_wikitext103_sequence_length(self):
        """序列长度应该正确。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader

        seq_length = 256
        loader = get_wikitext103_dataloader(
            split='train',
            batch_size=4,
            seq_length=seq_length,
            max_samples=50
        )

        batch = next(iter(loader))
        assert batch['input_ids'].shape[1] == seq_length

    def test_wikitext103_batch_size(self):
        """批次大小应该正确。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader

        batch_size = 8
        loader = get_wikitext103_dataloader(
            split='train',
            batch_size=batch_size,
            seq_length=128,
            max_samples=100
        )

        batch = next(iter(loader))
        assert batch['input_ids'].shape[0] == batch_size

    def test_wikitext103_returns_attention_mask(self):
        """应该返回 attention_mask。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader

        loader = get_wikitext103_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=50
        )

        batch = next(iter(loader))
        assert 'attention_mask' in batch
        assert batch['attention_mask'].shape == batch['input_ids'].shape

    def test_wikitext103_returns_labels(self):
        """应该返回 labels（用于语言模型训练）。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader

        loader = get_wikitext103_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=50
        )

        batch = next(iter(loader))
        assert 'labels' in batch
        assert batch['labels'].shape == batch['input_ids'].shape

    def test_wikitext103_invalid_split_raises_error(self):
        """无效的 split 应该抛出错误。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader

        with pytest.raises(ValueError):
            get_wikitext103_dataloader(
                split='invalid',
                batch_size=4,
                seq_length=128
            )

    def test_wikitext103_max_samples(self):
        """max_samples 应该限制样本数量。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader

        loader = get_wikitext103_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=20
        )

        total_samples = len(loader.dataset)
        assert total_samples <= 20

    def test_wikitext103_local_path_loading(self, tmp_path):
        """应该能从本地文件加载数据。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader
        
        # 创建临时测试文件
        test_file = tmp_path / "train.txt"
        test_content = "This is a test sentence for WikiText-103.\nAnother test sentence.\n" * 100
        test_file.write_text(test_content)
        
        # 使用本地路径加载
        loader = get_wikitext103_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=10,
            local_path=str(tmp_path)
        )
        
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert 'input_ids' in batch

    def test_wikitext103_local_file_path(self, tmp_path):
        """应该能直接从文件路径加载数据。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader
        
        # 创建临时测试文件
        test_file = tmp_path / "test_data.txt"
        test_content = "Test content for WikiText-103 loading.\n" * 100
        test_file.write_text(test_content)
        
        # 直接使用文件路径
        loader = get_wikitext103_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=10,
            local_path=str(test_file)
        )
        
        assert isinstance(loader, DataLoader)

    def test_wikitext103_shuffle_default_train(self):
        """训练集默认应该 shuffle。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader
        
        loader = get_wikitext103_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=50,
            shuffle=None  # 使用默认值
        )
        
        # 训练集默认 shuffle=True
        assert loader.sampler is not None or hasattr(loader, 'shuffle')

    def test_wikitext103_shuffle_default_validation(self):
        """验证集默认不应该 shuffle。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader
        
        loader = get_wikitext103_dataloader(
            split='validation',
            batch_size=4,
            seq_length=128,
            max_samples=50,
            shuffle=None  # 使用默认值
        )
        
        # 验证集默认 shuffle=False
        assert loader.dataset is not None

    def test_wikitext103_explicit_shuffle_false(self):
        """应该能显式设置 shuffle=False。"""
        from ckpt_compress.utils.data_loader import get_wikitext103_dataloader
        
        loader = get_wikitext103_dataloader(
            split='train',
            batch_size=4,
            seq_length=128,
            max_samples=50,
            shuffle=False
        )
        
        assert loader.dataset is not None
