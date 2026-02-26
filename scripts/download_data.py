"""
数据下载脚本。

提供下载 CIFAR-10、CIFAR-100、WikiText-2、WikiText-103 和 GLUE (SST-2, MNLI, STS-B) 数据集的功能。
"""

import os
import argparse
from typing import List, Optional, Dict
from pathlib import Path


def download_cifar10(data_dir: str = "./data") -> bool:
    """
    下载 CIFAR-10 数据集。
    
    参数:
        data_dir: 数据存储目录
        
    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 导入 torchvision
        from torchvision import datasets
        
        # 下载训练集和测试集
        print(f"Downloading CIFAR-10 to {data_dir}...")
        datasets.CIFAR10(root=data_dir, train=True, download=True)
        datasets.CIFAR10(root=data_dir, train=False, download=True)
        
        print("✓ CIFAR-10 downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download CIFAR-10: {e}")
        return False


def download_cifar100(data_dir: str = "./data") -> bool:
    """
    下载 CIFAR-100 数据集。
    
    参数:
        data_dir: 数据存储目录
        
    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 导入 torchvision
        from torchvision import datasets
        
        # 下载训练集和测试集
        print(f"Downloading CIFAR-100 to {data_dir}...")
        datasets.CIFAR100(root=data_dir, train=True, download=True)
        datasets.CIFAR100(root=data_dir, train=False, download=True)
        
        print("✓ CIFAR-100 downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download CIFAR-100: {e}")
        return False


def download_wikitext2(data_dir: str = "./data") -> bool:
    """
    下载 WikiText-2 数据集。
    
    参数:
        data_dir: 数据存储目录
        
    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 导入 datasets
        from datasets import load_dataset
        
        # 下载数据集
        print(f"Downloading WikiText-2 to {data_dir}...")
        load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=data_dir)
        
        print("✓ WikiText-2 downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download WikiText-2: {e}")
        return False


def download_wikitext103(data_dir: str = "./data") -> bool:
    """
    下载 WikiText-103 数据集。

    参数:
        data_dir: 数据存储目录

    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)

        # 导入 datasets
        from datasets import load_dataset

        # 下载数据集
        print(f"Downloading WikiText-103 to {data_dir}...")
        load_dataset('wikitext', 'wikitext-103-raw-v1', cache_dir=data_dir)

        print("✓ WikiText-103 downloaded successfully!")
        return True

    except Exception as e:
        print(f"✗ Failed to download WikiText-103: {e}")
        return False


def download_sst2(data_dir: str = "./data") -> bool:
    """
    下载 SST-2 (Stanford Sentiment Treebank) 数据集。

    SST-2 是 GLUE 基准测试的一部分，用于二分类情感分析任务。

    参数:
        data_dir: 数据存储目录

    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)

        # 导入 datasets
        from datasets import load_dataset

        # 禁用磁盘空间检查（某些环境下会有误报）
        import datasets.builder
        original_check = getattr(datasets.builder, 'has_sufficient_disk_space', None)
        if original_check:
            datasets.builder.has_sufficient_disk_space = lambda *args, **kwargs: True

        try:
            # 下载数据集
            print(f"Downloading SST-2 to {data_dir}...")
            dataset = load_dataset('glue', 'sst2', cache_dir=data_dir)
            print(f"✓ SST-2 downloaded successfully! (train: {len(dataset['train'])}, val: {len(dataset['validation'])})")
            return True
        finally:
            # 恢复原始函数
            if original_check:
                datasets.builder.has_sufficient_disk_space = original_check

    except Exception as e:
        print(f"✗ Failed to download SST-2: {e}")
        return False


def download_mnli(data_dir: str = "./data") -> bool:
    """
    下载 MNLI (Multi-Genre Natural Language Inference) 数据集。

    MNLI 是 GLUE 基准测试的一部分，用于自然语言推理任务。

    参数:
        data_dir: 数据存储目录

    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)

        # 导入 datasets
        from datasets import load_dataset

        # 禁用磁盘空间检查（某些环境下会有误报）
        import datasets.builder
        original_check = getattr(datasets.builder, 'has_sufficient_disk_space', None)
        if original_check:
            datasets.builder.has_sufficient_disk_space = lambda *args, **kwargs: True

        try:
            # 下载数据集
            print(f"Downloading MNLI to {data_dir}...")
            dataset = load_dataset('glue', 'mnli', cache_dir=data_dir)
            print(f"✓ MNLI downloaded successfully! (train: {len(dataset['train'])}, val_matched: {len(dataset['validation_matched'])}, val_mismatched: {len(dataset['validation_mismatched'])})")
            return True
        finally:
            # 恢复原始函数
            if original_check:
                datasets.builder.has_sufficient_disk_space = original_check

    except Exception as e:
        print(f"✗ Failed to download MNLI: {e}")
        return False


def download_stsb(data_dir: str = "./data") -> bool:
    """
    下载 STS-B (Semantic Textual Similarity Benchmark) 数据集。

    STS-B 是 GLUE 基准测试的一部分，用于语义相似度回归任务。

    参数:
        data_dir: 数据存储目录

    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)

        # 导入 datasets
        from datasets import load_dataset

        # 禁用磁盘空间检查（某些环境下会有误报）
        import datasets.builder
        original_check = getattr(datasets.builder, 'has_sufficient_disk_space', None)
        if original_check:
            datasets.builder.has_sufficient_disk_space = lambda *args, **kwargs: True

        try:
            # 下载数据集
            print(f"Downloading STS-B to {data_dir}...")
            dataset = load_dataset('glue', 'stsb', cache_dir=data_dir)
            print(f"✓ STS-B downloaded successfully! (train: {len(dataset['train'])}, val: {len(dataset['validation'])})")
            return True
        finally:
            # 恢复原始函数
            if original_check:
                datasets.builder.has_sufficient_disk_space = original_check

    except Exception as e:
        print(f"✗ Failed to download STS-B: {e}")
        return False


def download_all(
    data_dir: str = "./data",
    datasets: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    下载所有或指定的数据集。
    
    参数:
        data_dir: 数据存储目录
        datasets: 要下载的数据集列表，None 表示下载所有
        
    返回:
        Dict[str, bool]: 每个数据集的下载结果
    """
    # 可用的数据集
    available_datasets = {
        'cifar10': download_cifar10,
        'cifar100': download_cifar100,
        'wikitext2': download_wikitext2,
        'wikitext103': download_wikitext103,
        'sst2': download_sst2,
        'mnli': download_mnli,
        'stsb': download_stsb,
    }
    
    # 如果未指定数据集，下载所有
    if datasets is None:
        datasets = list(available_datasets.keys())
    
    # 下载每个数据集
    results = {}
    for dataset_name in datasets:
        if dataset_name in available_datasets:
            print(f"\n{'='*60}")
            print(f"Downloading {dataset_name}...")
            print(f"{'='*60}")
            results[dataset_name] = available_datasets[dataset_name](data_dir)
        else:
            print(f"✗ Unknown dataset: {dataset_name}")
            results[dataset_name] = False
    
    # 打印总结
    print(f"\n{'='*60}")
    print("Download Summary:")
    print(f"{'='*60}")
    for dataset_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {dataset_name}: {'Success' if success else 'Failed'}")
    
    return results


def verify_dataset(dataset_name: str, data_dir: str = "./data") -> bool:
    """
    验证数据集是否已下载。
    
    参数:
        dataset_name: 数据集名称
        data_dir: 数据存储目录
        
    返回:
        bool: 数据集存在返回 True，否则返回 False
        
    异常:
        ValueError: 如果数据集名称无效
    """
    data_path = Path(data_dir)
    
    # 检查不同数据集的存在性
    if dataset_name == 'cifar10':
        cifar10_path = data_path / "cifar-10-batches-py"
        return cifar10_path.exists()
    
    elif dataset_name == 'cifar100':
        cifar100_path = data_path / "cifar-100-python"
        return cifar100_path.exists()
    
    elif dataset_name == 'wikitext2':
        # WikiText-2 存储在 HuggingFace cache 中
        # 检查 cache 目录是否存在
        return data_path.exists()
    
    elif dataset_name == 'wikitext103':
        # WikiText-103 存储在 HuggingFace cache 中
        # 检查 cache 目录是否存在
        return data_path.exists()

    elif dataset_name in ('sst2', 'mnli', 'stsb'):
        # GLUE 数据集存储在 HuggingFace cache 中
        return data_path.exists()

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    """命令行接口。"""
    parser = argparse.ArgumentParser(
        description="Download datasets for ckpt-compress project"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset to download (cifar10, cifar100, wikitext2, wikitext103, sst2, mnli, stsb). '
             'Can specify multiple datasets separated by commas.'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all datasets'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Directory to store datasets (default: ./data)'
    )
    
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify if a dataset exists (cifar10, cifar100, wikitext2, wikitext103, sst2, mnli, stsb)'
    )
    
    args = parser.parse_args()
    
    # 验证数据集
    if args.verify:
        exists = verify_dataset(args.verify, args.data_dir)
        if exists:
            print(f"✓ {args.verify} exists in {args.data_dir}")
        else:
            print(f"✗ {args.verify} not found in {args.data_dir}")
        return
    
    # 下载数据集
    if args.all:
        download_all(data_dir=args.data_dir)
    elif args.dataset:
        datasets = [d.strip() for d in args.dataset.split(',')]
        download_all(data_dir=args.data_dir, datasets=datasets)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
