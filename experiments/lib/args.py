"""共享 argparse 参数定义。

将 8 个 run_*.py 脚本中重复的参数定义提取到此处，
各脚本按需组合调用 add_*_args() 函数。
"""

import argparse


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """创建包含所有通用参数的 parser。"""
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """所有脚本都需要的基础参数（8/8 脚本共享）。"""
    parser.add_argument('--model', type=str, required=True, help='模型名称')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--checkpoint', type=str, default=None, help='微调检查点路径')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=None)


def add_scoring_args(parser: argparse.ArgumentParser) -> None:
    """重要性得分计算相关参数（7/8 脚本共享，run_fault_tolerant 不用）。"""
    parser.add_argument('--alpha', type=float, default=0.5, help='二阶项权重')
    parser.add_argument('--num_steps', type=int, default=100, help='梯度累积步数')
    parser.add_argument('--hvp_batches', type=int, default=8)
    parser.add_argument('--hvp_mode', type=str, default='full', choices=['full', 'block'])
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--eval_batches', type=int, default=20, help='评估批次数')


def add_prune_ratios_arg(parser: argparse.ArgumentParser,
                         default: str = '0.1,0.2,0.3,0.4') -> None:
    """添加 --prune_ratios（逗号分隔的字符串）。"""
    parser.add_argument('--prune_ratios', type=str, default=default)


def parse_prune_ratios(args) -> list:
    """将 args.prune_ratios 字符串解析为 float 列表。"""
    return [float(r) for r in args.prune_ratios.split(',')]
