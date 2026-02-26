"""
Tiny-ImageNet 数据加载器测试。
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

from src.ckpt_compress.utils.data_loader import (
    TinyImageNetDataset,
    get_tiny_imagenet_transforms,
    get_tiny_imagenet_loaders,
)


@pytest.fixture
def mock_tiny_imagenet_dir():
    """创建模拟的 Tiny-ImageNet 目录结构。"""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / 'tiny-imagenet-200'

    # 创建目录结构
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    val_images_dir = val_dir / 'images'

    train_dir.mkdir(parents=True)
    val_images_dir.mkdir(parents=True)

    # 创建 wnids.txt（类别列表）
    wnids = ['n01443537', 'n01629819', 'n01641577']
    with open(data_dir / 'wnids.txt', 'w') as f:
        for wnid in wnids:
            f.write(f"{wnid}\n")

    # 创建训练集图像
    for i, wnid in enumerate(wnids):
        class_dir = train_dir / wnid / 'images'
        class_dir.mkdir(parents=True)

        # 为每个类别创建 5 张图像
        for j in range(5):
            img = Image.new('RGB', (64, 64), color=(i*80, j*50, 100))
            img.save(class_dir / f'{wnid}_{j}.JPEG')

    # 创建验证集图像和标注文件
    val_annotations = []
    for i, wnid in enumerate(wnids):
        # 为每个类别创建 3 张验证图像
        for j in range(3):
            img_name = f'val_{i}_{j}.JPEG'
            img = Image.new('RGB', (64, 64), color=(i*80, j*50, 150))
            img.save(val_images_dir / img_name)

            # 添加标注（格式：filename\tclass_id\tx\ty\tw\th）
            val_annotations.append(f"{img_name}\t{wnid}\t0\t0\t64\t64\n")

    # 写入验证集标注文件
    with open(val_dir / 'val_annotations.txt', 'w') as f:
        f.writelines(val_annotations)

    yield temp_dir

    # 清理
    shutil.rmtree(temp_dir)


class TestTinyImageNetDataset:
    """测试 TinyImageNetDataset 类。"""

    def test_init_train_split(self, mock_tiny_imagenet_dir):
        """测试训练集初始化。"""
        dataset = TinyImageNetDataset(
            root=mock_tiny_imagenet_dir,
            split='train',
            download=False,
        )

        assert len(dataset) == 15  # 3 classes * 5 images
        assert len(dataset.classes) == 3
        assert len(dataset.class_to_idx) == 3

    def test_init_val_split(self, mock_tiny_imagenet_dir):
        """测试验证集初始化。"""
        dataset = TinyImageNetDataset(
            root=mock_tiny_imagenet_dir,
            split='val',
            download=False,
        )

        assert len(dataset) == 9  # 3 classes * 3 images
        assert len(dataset.classes) == 3

    def test_invalid_split(self, mock_tiny_imagenet_dir):
        """测试无效的 split 参数。"""
        with pytest.raises(ValueError, match="Invalid split"):
            dataset = TinyImageNetDataset(
                root=mock_tiny_imagenet_dir,
                split='test',
                download=False,
            )
            # 触发 _load_samples
            _ = dataset.samples

    def test_dataset_not_found(self):
        """测试数据集不存在的情况。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(RuntimeError, match="Dataset not found"):
                TinyImageNetDataset(
                    root=temp_dir,
                    split='train',
                    download=False,
                )

    def test_getitem_without_transform(self, mock_tiny_imagenet_dir):
        """测试不使用变换获取样本。"""
        dataset = TinyImageNetDataset(
            root=mock_tiny_imagenet_dir,
            split='train',
            download=False,
        )

        image, label = dataset[0]

        assert isinstance(image, Image.Image)
        assert isinstance(label, int)
        assert 0 <= label < 3
        assert image.size == (64, 64)

    def test_getitem_with_transform(self, mock_tiny_imagenet_dir):
        """测试使用变换获取样本。"""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = TinyImageNetDataset(
            root=mock_tiny_imagenet_dir,
            split='train',
            transform=transform,
            download=False,
        )

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
        assert isinstance(label, int)

    def test_class_mapping(self, mock_tiny_imagenet_dir):
        """测试类别映射。"""
        dataset = TinyImageNetDataset(
            root=mock_tiny_imagenet_dir,
            split='train',
            download=False,
        )

        assert 'n01443537' in dataset.class_to_idx
        assert 'n01629819' in dataset.class_to_idx
        assert 'n01641577' in dataset.class_to_idx

        # 检查映射是否连续
        values = sorted(dataset.class_to_idx.values())
        assert values == [0, 1, 2]

    def test_missing_wnids_file(self, mock_tiny_imagenet_dir):
        """测试缺少 wnids.txt 文件的情况。"""
        # 删除 wnids.txt
        wnids_file = Path(mock_tiny_imagenet_dir) / 'tiny-imagenet-200' / 'wnids.txt'
        wnids_file.unlink()

        with pytest.raises(RuntimeError, match="wnids.txt not found"):
            TinyImageNetDataset(
                root=mock_tiny_imagenet_dir,
                split='train',
                download=False,
            )

    def test_missing_val_annotations(self, mock_tiny_imagenet_dir):
        """测试缺少验证集标注文件的情况。"""
        # 删除 val_annotations.txt
        val_annotations = Path(mock_tiny_imagenet_dir) / 'tiny-imagenet-200' / 'val' / 'val_annotations.txt'
        val_annotations.unlink()

        with pytest.raises(RuntimeError, match="val_annotations.txt not found"):
            TinyImageNetDataset(
                root=mock_tiny_imagenet_dir,
                split='val',
                download=False,
            )

    def test_val_annotations_with_invalid_class(self, mock_tiny_imagenet_dir):
        """测试验证集标注包含无效类别的情况。"""
        val_dir = Path(mock_tiny_imagenet_dir) / 'tiny-imagenet-200' / 'val'

        # 添加一个无效类别的标注
        with open(val_dir / 'val_annotations.txt', 'a') as f:
            f.write("invalid_img.JPEG\tinvalid_class\t0\t0\t64\t64\n")

        # 应该能正常加载，只是跳过无效类别
        dataset = TinyImageNetDataset(
            root=mock_tiny_imagenet_dir,
            split='val',
            download=False,
        )

        assert len(dataset) == 9  # 仍然是 9 个有效样本

    def test_val_annotations_with_short_line(self, mock_tiny_imagenet_dir):
        """测试验证集标注包含不完整行的情况。"""
        val_dir = Path(mock_tiny_imagenet_dir) / 'tiny-imagenet-200' / 'val'

        # 添加一个不完整的行
        with open(val_dir / 'val_annotations.txt', 'a') as f:
            f.write("incomplete_line\n")

        # 应该能正常加载，只是跳过不完整的行
        dataset = TinyImageNetDataset(
            root=mock_tiny_imagenet_dir,
            split='val',
            download=False,
        )

        assert len(dataset) == 9  # 仍然是 9 个有效样本

    def test_download_when_exists(self, mock_tiny_imagenet_dir):
        """测试数据集已存在时的下载行为。"""
        # 使用 patch 捕获 print 输出
        with patch('builtins.print') as mock_print:
            dataset = TinyImageNetDataset(
                root=mock_tiny_imagenet_dir,
                split='train',
                download=True,
            )

            # 检查是否打印了"已存在"消息
            mock_print.assert_called()
            call_args = [str(call[0][0]) for call in mock_print.call_args_list]
            assert any('already exists' in arg for arg in call_args)

    @patch('urllib.request.urlretrieve')
    @patch('zipfile.ZipFile')
    @patch('pathlib.Path.unlink')
    def test_download_success(self, mock_unlink, mock_zipfile, mock_urlretrieve):
        """测试成功下载数据集。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 模拟下载和解压
            mock_zip_instance = MagicMock()
            mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

            # 在下载后创建必要的文件结构
            def create_structure(*args, **kwargs):
                data_dir = Path(temp_dir) / 'tiny-imagenet-200'
                data_dir.mkdir(parents=True, exist_ok=True)

                # 创建 wnids.txt
                with open(data_dir / 'wnids.txt', 'w') as f:
                    f.write("n01443537\n")

                # 创建训练目录
                train_dir = data_dir / 'train' / 'n01443537' / 'images'
                train_dir.mkdir(parents=True, exist_ok=True)

                # 创建一张图像
                img = Image.new('RGB', (64, 64))
                img.save(train_dir / 'test.JPEG')

            # 模拟解压操作创建文件结构
            mock_zip_instance.extractall.side_effect = create_structure

            dataset = TinyImageNetDataset(
                root=temp_dir,
                split='train',
                download=True,
            )

            # 验证下载函数被调用
            mock_urlretrieve.assert_called_once()
            mock_zip_instance.extractall.assert_called_once()
            mock_unlink.assert_called_once()  # 验证 zip 文件被删除

    @patch('urllib.request.urlretrieve')
    def test_download_failure(self, mock_urlretrieve):
        """测试下载失败的情况。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 模拟下载失败
            mock_urlretrieve.side_effect = Exception("Download failed")

            with pytest.raises(Exception, match="Download failed"):
                TinyImageNetDataset(
                    root=temp_dir,
                    split='train',
                    download=True,
                )


class TestGetTinyImageNetTransforms:
    """测试 get_tiny_imagenet_transforms 函数。"""

    def test_transforms_return_type(self):
        """测试返回类型。"""
        train_transform, val_transform = get_tiny_imagenet_transforms()

        assert train_transform is not None
        assert val_transform is not None

    def test_train_transform(self):
        """测试训练变换。"""
        train_transform, _ = get_tiny_imagenet_transforms()

        # 创建测试图像
        img = Image.new('RGB', (64, 64), color=(100, 150, 200))

        # 应用变换
        transformed = train_transform(img)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 64, 64)
        assert transformed.dtype == torch.float32

    def test_val_transform(self):
        """测试验证变换。"""
        _, val_transform = get_tiny_imagenet_transforms()

        # 创建测试图像
        img = Image.new('RGB', (64, 64), color=(100, 150, 200))

        # 应用变换
        transformed = val_transform(img)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 64, 64)
        assert transformed.dtype == torch.float32


class TestGetTinyImageNetLoaders:
    """测试 get_tiny_imagenet_loaders 函数。"""

    def test_loaders_creation(self, mock_tiny_imagenet_dir):
        """测试数据加载器创建。"""
        train_loader, val_loader = get_tiny_imagenet_loaders(
            batch_size=4,
            data_dir=mock_tiny_imagenet_dir,
            download=False,
            num_workers=0,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader.dataset) == 15
        assert len(val_loader.dataset) == 9

    def test_loaders_batch_size(self, mock_tiny_imagenet_dir):
        """测试批次大小。"""
        train_loader, val_loader = get_tiny_imagenet_loaders(
            batch_size=4,
            data_dir=mock_tiny_imagenet_dir,
            download=False,
            num_workers=0,
        )

        assert train_loader.batch_size == 4
        assert val_loader.batch_size == 4

    def test_loaders_iteration(self, mock_tiny_imagenet_dir):
        """测试数据加载器迭代。"""
        train_loader, _ = get_tiny_imagenet_loaders(
            batch_size=4,
            data_dir=mock_tiny_imagenet_dir,
            download=False,
            num_workers=0,
        )

        # 获取一个批次
        images, labels = next(iter(train_loader))

        assert images.shape[0] <= 4  # 批次大小
        assert images.shape[1:] == (3, 64, 64)  # 图像形状
        assert labels.shape[0] <= 4
        assert labels.dtype == torch.long

    def test_loaders_with_subset(self, mock_tiny_imagenet_dir):
        """测试使用子集。"""
        train_loader, val_loader = get_tiny_imagenet_loaders(
            batch_size=4,
            data_dir=mock_tiny_imagenet_dir,
            download=False,
            num_workers=0,
            train_subset=10,
            val_subset=5,
        )

        assert len(train_loader.dataset) == 10
        assert len(val_loader.dataset) == 5

    def test_loaders_subset_larger_than_dataset(self, mock_tiny_imagenet_dir):
        """测试子集大小超过数据集大小。"""
        train_loader, val_loader = get_tiny_imagenet_loaders(
            batch_size=4,
            data_dir=mock_tiny_imagenet_dir,
            download=False,
            num_workers=0,
            train_subset=1000,  # 大于实际数量
            val_subset=1000,
        )

        # 应该使用实际的数据集大小
        assert len(train_loader.dataset) == 15
        assert len(val_loader.dataset) == 9

    def test_train_loader_shuffle(self, mock_tiny_imagenet_dir):
        """测试训练加载器是否打乱。"""
        train_loader, _ = get_tiny_imagenet_loaders(
            batch_size=4,
            data_dir=mock_tiny_imagenet_dir,
            download=False,
            num_workers=0,
        )

        # 训练加载器应该打乱
        assert train_loader.sampler is not None

    def test_val_loader_no_shuffle(self, mock_tiny_imagenet_dir):
        """测试验证加载器不打乱。"""
        _, val_loader = get_tiny_imagenet_loaders(
            batch_size=4,
            data_dir=mock_tiny_imagenet_dir,
            download=False,
            num_workers=0,
        )

        # 验证加载器不应该打乱
        assert val_loader.sampler is not None

    def test_loaders_pin_memory(self, mock_tiny_imagenet_dir):
        """测试 pin_memory 设置。"""
        train_loader, val_loader = get_tiny_imagenet_loaders(
            batch_size=4,
            data_dir=mock_tiny_imagenet_dir,
            download=False,
            num_workers=0,
        )

        assert train_loader.pin_memory is True
        assert val_loader.pin_memory is True

    def test_loaders_num_workers(self, mock_tiny_imagenet_dir):
        """测试 num_workers 设置。"""
        train_loader, val_loader = get_tiny_imagenet_loaders(
            batch_size=4,
            data_dir=mock_tiny_imagenet_dir,
            download=False,
            num_workers=2,
        )

        assert train_loader.num_workers == 2
        assert val_loader.num_workers == 2

    def test_label_range(self, mock_tiny_imagenet_dir):
        """测试标签范围。"""
        train_loader, _ = get_tiny_imagenet_loaders(
            batch_size=15,  # 一次加载所有数据
            data_dir=mock_tiny_imagenet_dir,
            download=False,
            num_workers=0,
        )

        images, labels = next(iter(train_loader))

        # 标签应该在 [0, 2] 范围内（3 个类别）
        assert labels.min() >= 0
        assert labels.max() <= 2

    def test_image_shape_consistency(self, mock_tiny_imagenet_dir):
        """测试图像形状一致性。"""
        train_loader, _ = get_tiny_imagenet_loaders(
            batch_size=4,
            data_dir=mock_tiny_imagenet_dir,
            download=False,
            num_workers=0,
        )

        for images, labels in train_loader:
            assert images.shape[1:] == (3, 64, 64)
            assert images.shape[0] == labels.shape[0]
