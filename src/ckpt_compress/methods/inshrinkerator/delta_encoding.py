"""
Inshrinkerator 增量编码模块。

实现量化感知的增量编码:
- 模运算增量计算: D = (prev - curr) mod B
- 按前一桶重排参数（算法 5/6）
- 游程编码（RLE）压缩
"""

import torch
from typing import List, Dict


def delta_encode(
    prev: torch.Tensor,
    curr: torch.Tensor,
    B: int
) -> torch.Tensor:
    """
    计算增量编码: D = (prev - curr) mod B。

    参数:
        prev: 前一量化索引
        curr: 当前量化索引
        B: 量化桶数

    返回:
        增量张量
    """
    D = (prev.long() - curr.long()) % B
    return D


def delta_decode(
    prev: torch.Tensor,
    D: torch.Tensor,
    B: int
) -> torch.Tensor:
    """
    解码增量以恢复当前索引: curr = (prev - D) mod B。

    参数:
        prev: 前一量化索引
        D: 增量张量
        B: 量化桶数

    返回:
        恢复的当前索引
    """
    curr = (prev.long() - D.long()) % B
    return curr


def rearrange_by_prev_bin(
    q_prev: torch.Tensor,
    D: torch.Tensor,
    B: int
) -> List[torch.Tensor]:
    """
    按前一桶重排增量以获得更好的 RLE 压缩。

    Inshrinkerator 论文的算法 5。

    参数:
        q_prev: 前一量化索引
        D: 增量张量
        B: 桶数

    返回:
        增量张量列表，每个桶一个
    """
    grouped = []
    for bin_idx in range(B):
        mask = (q_prev == bin_idx)
        grouped.append(D[mask])
    return grouped


def restore_rearrangement(
    q_prev: torch.Tensor,
    grouped: List[torch.Tensor],
    B: int
) -> torch.Tensor:
    """
    从分组增量恢复原始增量顺序。

    Inshrinkerator 论文的算法 6。

    参数:
        q_prev: 前一量化索引
        grouped: 每个桶的增量张量列表
        B: 桶数

    返回:
        原始顺序的恢复增量张量
    """
    D = torch.zeros_like(q_prev)

    # 跟踪每组内的位置
    group_positions = [0] * B

    for i in range(len(q_prev)):
        bin_idx = q_prev[i].item()
        pos = group_positions[bin_idx]
        D[i] = grouped[bin_idx][pos]
        group_positions[bin_idx] += 1

    return D


def rle_encode(D: torch.Tensor) -> bytes:
    """
    对增量张量进行游程编码。

    格式: 对于每个游程:
    - 如果游程长度 == 1: 只存储值
    - 如果游程长度 > 1: 存储 (值, -游程长度)
      使用负数来区分游程长度和值

    参数:
        D: 增量张量（一维，整数值）

    返回:
        编码后的字节
    """
    if D.numel() == 0:
        return b''

    D = D.flatten().tolist()
    encoded = []

    i = 0
    while i < len(D):
        value = D[i]
        run_length = 1

        # 计算连续相同值的数量
        while i + run_length < len(D) and D[i + run_length] == value:
            run_length += 1

        if run_length > 1:
            # 存储值和负游程长度
            encoded.append(value)
            encoded.append(-run_length)
        else:
            # 单个值，直接存储
            encoded.append(value)

        i += run_length

    # 转换为字节（为简单起见使用变长编码）
    # 目前使用简单的 int16 编码
    import struct
    result = struct.pack(f'{len(encoded)}h', *encoded)
    return result


def rle_decode(encoded: bytes, original_length: int) -> torch.Tensor:
    """
    解码游程编码数据。

    参数:
        encoded: RLE 编码的字节
        original_length: 原始张量长度

    返回:
        解码后的张量
    """
    if len(encoded) == 0 or original_length == 0:
        return torch.tensor([], dtype=torch.long)

    import struct
    n_values = len(encoded) // 2  # int16 = 2 字节
    values = list(struct.unpack(f'{n_values}h', encoded))

    decoded = []
    i = 0
    while i < len(values):
        value = values[i]
        i += 1

        # 检查下一个值是否为负游程长度
        if i < len(values) and values[i] < 0:
            run_length = -values[i]
            i += 1
            decoded.extend([value] * run_length)
        else:
            decoded.append(value)

    # 确保恰好有 original_length 个元素
    decoded = decoded[:original_length]
    while len(decoded) < original_length:
        decoded.append(0)

    return torch.tensor(decoded, dtype=torch.long)
