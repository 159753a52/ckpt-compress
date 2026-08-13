"""共享 argparse 参数定义。

将 8 个 run_*.py 脚本中重复的参数定义提取到此处，
各脚本按需组合调用 add_*_args() 函数。
"""

import argparse
import math
import re
from typing import Any


def positive_int(value: str) -> int:
    """Argparse type for strictly positive counts."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def nonnegative_int(value: str) -> int:
    """Argparse type for counts that may be zero."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return parsed


def unit_interval_float(value: str) -> float:
    """Argparse type for finite values in the closed unit interval."""
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"expected a finite value in [0, 1], got {value}")
    return parsed


def device_name(value: str) -> str:
    """Accept the CPU and CUDA device spellings supported by experiment code."""
    if value == "cpu" or re.fullmatch(r"cuda(?::[0-9]+)?", value):
        return value
    raise argparse.ArgumentTypeError(f"expected 'cpu', 'cuda', or 'cuda:<index>', got {value!r}")


def _parse_prune_ratio_text(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("prune ratios must be a non-empty comma-separated list")
    return [unit_interval_float(part) for part in parts]


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """创建包含所有通用参数的 parser。"""
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """所有脚本都需要的基础参数（8/8 脚本共享）。"""
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--checkpoint", type=str, default=None, help="微调检查点路径")
    parser.add_argument("--batch_size", type=positive_int, default=4)
    parser.add_argument("--seq_length", type=positive_int, default=512)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_workers", type=nonnegative_int, default=0)
    parser.add_argument("--device", type=device_name, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)


def add_scoring_args(parser: argparse.ArgumentParser) -> None:
    """重要性得分计算相关参数（7/8 脚本共享，run_fault_tolerant 不用）。"""
    parser.add_argument("--alpha", type=unit_interval_float, default=0.5, help="二阶项权重")
    parser.add_argument("--num_steps", type=positive_int, default=100, help="梯度累积步数")
    parser.add_argument("--hvp_batches", type=positive_int, default=8)
    parser.add_argument("--hvp_mode", type=str, default="full", choices=["full", "block"])
    parser.add_argument("--chunk_size", type=positive_int, default=10)
    parser.add_argument("--eval_batches", type=positive_int, default=20, help="评估批次数")


def add_prune_ratios_arg(
    parser: argparse.ArgumentParser,
    default: str = "0.1,0.2,0.3,0.4",
) -> None:
    """添加 --prune_ratios（逗号分隔的字符串）。"""
    parser.add_argument(
        "--prune_ratios",
        type=_parse_prune_ratio_text,
        default=_parse_prune_ratio_text(default),
    )


def parse_prune_ratios(args: Any) -> list[float]:
    """Read and validate ratios from a namespace or a raw CLI value."""
    value = getattr(args, "prune_ratios", args)
    if isinstance(value, str):
        return _parse_prune_ratio_text(value)
    if not isinstance(value, (list, tuple)):
        raise TypeError("prune_ratios must be a comma-separated string or a sequence")
    try:
        return [unit_interval_float(str(ratio)) for ratio in value]
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc
