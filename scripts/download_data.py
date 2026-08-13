"""
数据下载脚本。

提供下载 CIFAR-10、CIFAR-100、WikiText-2、WikiText-103 和 GLUE (SST-2, MNLI, STS-B) 数据集的功能。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATASET_NAMES = ("cifar10", "cifar100", "wikitext2", "wikitext103", "sst2", "mnli", "stsb")
DatasetDownloader = Callable[[str], bool]


@dataclass(frozen=True)
class HFDatasetSpec:
    """Cache metadata needed to identify one downloaded Hugging Face dataset."""

    builder_name: str
    config_name: str
    required_splits: tuple[str, ...]


HF_DATASET_SPECS = {
    "wikitext2": HFDatasetSpec("wikitext", "wikitext-2-raw-v1", ("train", "validation", "test")),
    "wikitext103": HFDatasetSpec(
        "wikitext", "wikitext-103-raw-v1", ("train", "validation", "test")
    ),
    "sst2": HFDatasetSpec("glue", "sst2", ("train", "validation")),
    "mnli": HFDatasetSpec("glue", "mnli", ("train", "validation_matched", "validation_mismatched")),
    "stsb": HFDatasetSpec("glue", "stsb", ("train", "validation")),
}

CIFAR_DATASET_FILES = {
    "cifar10": (
        "cifar-10-batches-py",
        (
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
            "test_batch",
            "batches.meta",
        ),
    ),
    "cifar100": (
        "cifar-100-python",
        ("train", "test", "meta"),
    ),
}


def _parse_selection(value: str, option_name: str, available: Sequence[str]) -> list[str]:
    """Parse and validate a comma-separated CLI selection."""
    selections = [item.strip() for item in value.split(",")]
    if not value.strip() or any(not item for item in selections):
        raise argparse.ArgumentTypeError(
            f"{option_name} must contain one or more comma-separated choices"
        )

    unknown = [item for item in selections if item not in available]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {option_name}: {', '.join(unknown)}; " f"choose from {', '.join(available)}"
        )
    return selections


def _parse_dataset_selection(value: str) -> list[str]:
    return _parse_selection(value, "dataset", DATASET_NAMES)


def _parse_verify_dataset(value: str) -> str:
    if not value.strip() or "," in value:
        raise argparse.ArgumentTypeError("verify expects exactly one dataset name")
    if value not in DATASET_NAMES:
        raise argparse.ArgumentTypeError(
            f"unknown dataset: {value}; choose from {', '.join(DATASET_NAMES)}"
        )
    return value


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
        load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=data_dir)

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
        load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=data_dir)

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
        # 禁用磁盘空间检查（某些环境下会有误报）
        import datasets.builder
        from datasets import load_dataset

        original_check = getattr(datasets.builder, "has_sufficient_disk_space", None)
        if original_check:
            datasets.builder.has_sufficient_disk_space = lambda *args, **kwargs: True

        try:
            # 下载数据集
            print(f"Downloading SST-2 to {data_dir}...")
            dataset = load_dataset("glue", "sst2", cache_dir=data_dir)
            print(
                f"✓ SST-2 downloaded successfully! (train: {len(dataset['train'])}, val: {len(dataset['validation'])})"
            )
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
        # 禁用磁盘空间检查（某些环境下会有误报）
        import datasets.builder
        from datasets import load_dataset

        original_check = getattr(datasets.builder, "has_sufficient_disk_space", None)
        if original_check:
            datasets.builder.has_sufficient_disk_space = lambda *args, **kwargs: True

        try:
            # 下载数据集
            print(f"Downloading MNLI to {data_dir}...")
            dataset = load_dataset("glue", "mnli", cache_dir=data_dir)
            print(
                f"✓ MNLI downloaded successfully! (train: {len(dataset['train'])}, val_matched: {len(dataset['validation_matched'])}, val_mismatched: {len(dataset['validation_mismatched'])})"
            )
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
        # 禁用磁盘空间检查（某些环境下会有误报）
        import datasets.builder
        from datasets import load_dataset

        original_check = getattr(datasets.builder, "has_sufficient_disk_space", None)
        if original_check:
            datasets.builder.has_sufficient_disk_space = lambda *args, **kwargs: True

        try:
            # 下载数据集
            print(f"Downloading STS-B to {data_dir}...")
            dataset = load_dataset("glue", "stsb", cache_dir=data_dir)
            print(
                f"✓ STS-B downloaded successfully! (train: {len(dataset['train'])}, val: {len(dataset['validation'])})"
            )
            return True
        finally:
            # 恢复原始函数
            if original_check:
                datasets.builder.has_sufficient_disk_space = original_check

    except Exception as e:
        print(f"✗ Failed to download STS-B: {e}")
        return False


def download_all(data_dir: str = "./data", datasets: Optional[list[str]] = None) -> dict[str, bool]:
    """
    下载所有或指定的数据集。

    参数:
        data_dir: 数据存储目录
        datasets: 要下载的数据集列表，None 表示下载所有

    返回:
        Dict[str, bool]: 每个数据集的下载结果
    """
    # 可用的数据集
    available_datasets: dict[str, DatasetDownloader] = {
        "cifar10": download_cifar10,
        "cifar100": download_cifar100,
        "wikitext2": download_wikitext2,
        "wikitext103": download_wikitext103,
        "sst2": download_sst2,
        "mnli": download_mnli,
        "stsb": download_stsb,
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


def _non_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _verify_cifar_dataset(dataset_name: str, data_dir: Path) -> bool:
    """Verify the official CIFAR directory and its required payload files."""
    directory_name, required_files = CIFAR_DATASET_FILES[dataset_name]
    dataset_path = data_dir / directory_name
    return dataset_path.is_dir() and all(
        _non_empty_file(dataset_path / filename) for filename in required_files
    )


def _safe_relative_file(directory: Path, filename: str) -> Optional[Path]:
    """Resolve a cache metadata filename without allowing path traversal."""
    path = (directory / filename).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError:
        return None
    return path


def _metadata_files_for_split(cache_path: Path, split: str, split_info: object) -> list[Path]:
    """Return files explicitly associated with a split by cache metadata."""
    filenames: list[str] = []
    if isinstance(split_info, dict):
        filename = split_info.get("filename")
        if isinstance(filename, str):
            filenames.append(filename)
        data_files = split_info.get("data_files")
        if isinstance(data_files, list):
            for item in data_files:
                if isinstance(item, dict):
                    filename = item.get("filename")
                    if isinstance(filename, str):
                        filenames.append(filename)

    state_path = cache_path / "state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = None
        if isinstance(state, dict):
            data_files = state.get("_data_files")
            if isinstance(data_files, list):
                for item in data_files:
                    if isinstance(item, dict) and isinstance(item.get("filename"), str):
                        filename = item["filename"]
                        if split.casefold() in Path(filename).name.casefold():
                            filenames.append(filename)

    resolved = []
    for filename in filenames:
        path = _safe_relative_file(cache_path, filename)
        if path is not None and path not in resolved:
            resolved.append(path)
    return resolved


def _conventional_files_for_split(cache_path: Path, split: str) -> list[Path]:
    """Find the Arrow/Parquet files used by the standard datasets cache layout."""
    split_token = re.compile(rf"(?:^|[-_.]){re.escape(split.casefold())}(?:[-_.]|$)")
    return [
        path
        for path in cache_path.iterdir()
        if path.is_file()
        and path.suffix.casefold() in {".arrow", ".parquet"}
        and split_token.search(path.stem.casefold()) is not None
    ]


def _verify_hf_cache(cache_dir: Path, spec: HFDatasetSpec) -> bool:
    """Verify matching Hugging Face metadata and completed split payloads offline."""
    if not cache_dir.is_dir():
        return False

    for info_path in cache_dir.rglob("dataset_info.json"):
        try:
            metadata = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(metadata, dict):
            continue
        if (
            metadata.get("builder_name") != spec.builder_name
            or metadata.get("config_name") != spec.config_name
        ):
            continue

        splits = metadata.get("splits")
        if not isinstance(splits, dict):
            continue
        if any(split not in splits for split in spec.required_splits):
            continue

        cache_path = info_path.parent
        complete = True
        for split in spec.required_splits:
            split_info = splits[split]
            if not isinstance(split_info, dict):
                complete = False
                break
            if split_info.get("num_examples", 0) <= 0 or split_info.get("num_bytes", 0) <= 0:
                complete = False
                break

            files = _metadata_files_for_split(cache_path, split, split_info)
            if not files:
                files = _conventional_files_for_split(cache_path, split)
            if not files or not all(_non_empty_file(path) for path in files):
                complete = False
                break

        if complete:
            return True

    return False


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
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_path = Path(data_dir)
    if dataset_name in CIFAR_DATASET_FILES:
        return _verify_cifar_dataset(dataset_name, data_path)
    return _verify_hf_cache(data_path, HF_DATASET_SPECS[dataset_name])


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行接口。"""
    parser = argparse.ArgumentParser(description="Download datasets for ckpt-compress project")

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--dataset",
        type=_parse_dataset_selection,
        help="Dataset to download (cifar10, cifar100, wikitext2, wikitext103, sst2, mnli, stsb). "
        "Can specify multiple datasets separated by commas.",
    )

    selection.add_argument("--all", action="store_true", help="Download all datasets")

    selection.add_argument(
        "--verify",
        type=_parse_verify_dataset,
        help="Verify if a dataset exists (cifar10, cifar100, wikitext2, wikitext103, sst2, mnli, stsb)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to store datasets (default: ./data)",
    )

    args = parser.parse_args(argv)

    # 验证数据集
    if args.verify is not None:
        exists = verify_dataset(args.verify, args.data_dir)
        if exists:
            print(f"✓ {args.verify} exists in {args.data_dir}")
        else:
            print(f"✗ {args.verify} not found in {args.data_dir}")
        return 0 if exists else 1

    # 下载数据集
    if args.all:
        results = download_all(data_dir=args.data_dir)
    else:
        results = download_all(data_dir=args.data_dir, datasets=args.dataset)
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
