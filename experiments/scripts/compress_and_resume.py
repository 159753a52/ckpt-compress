"""
检查点压缩和恢复训练脚本。

支持使用不同的压缩方法压缩检查点，并从压缩的检查点恢复训练。
"""

import argparse
import torch
import time
from pathlib import Path
from typing import Dict, Any
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.methods.excp.excp import ExCPCompressor
from src.ckpt_compress.methods.inshrinkerator.inshrinkerator import InshrinkeratorCompressor
from src.ckpt_compress.methods.predictive.predictive import PredictiveResidualCompressor


def get_compressor(method: str):
    """
    获取压缩器。

    参数:
        method: 压缩方法名称

    返回:
        压缩器实例
    """
    if method == 'excp':
        return ExCPCompressor()
    elif method == 'inshrinkerator':
        return InshrinkeratorCompressor()
    elif method == 'predictive':
        return PredictiveResidualCompressor()
    else:
        raise ValueError(f"Unknown compression method: {method}")


def compress_checkpoint(
    checkpoint_path: str,
    method: str,
    output_path: str,
) -> Dict[str, Any]:
    """
    压缩检查点。

    参数:
        checkpoint_path: 检查点文件路径
        method: 压缩方法
        output_path: 输出文件路径

    返回:
        包含压缩统计信息的字典
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 提取模型状态字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    print(f"Checkpoint contains {len(state_dict)} tensors")

    # 计算原始大小
    original_size = sum(
        tensor.element_size() * tensor.nelement()
        for tensor in state_dict.values()
    )
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")

    # 压缩
    print(f"Compressing with method: {method}")
    compressor = get_compressor(method)

    start_time = time.time()
    compressed_data = compressor.compress(state_dict)
    compression_time = time.time() - start_time

    compressed_size = len(compressed_data)
    compression_ratio = original_size / compressed_size

    print(f"Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Compression time: {compression_time:.2f}s")

    # 保存压缩数据和元数据
    output_data = {
        'compressed_data': compressed_data,
        'method': method,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'compression_time': compression_time,
    }

    # 如果原始检查点包含其他信息（如优化器状态），也保存
    if 'optimizer_state_dict' in checkpoint:
        output_data['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
    if 'epoch' in checkpoint:
        output_data['epoch'] = checkpoint['epoch']
    if 'history' in checkpoint:
        output_data['history'] = checkpoint['history']

    print(f"Saving compressed checkpoint to: {output_path}")
    torch.save(output_data, output_path)

    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'compression_time': compression_time,
    }


def decompress_checkpoint(
    compressed_checkpoint_path: str,
    output_path: str = None,
) -> Dict[str, Any]:
    """
    解压检查点。

    参数:
        compressed_checkpoint_path: 压缩检查点文件路径
        output_path: 输出文件路径（可选）

    返回:
        解压后的检查点字典
    """
    print(f"Loading compressed checkpoint from: {compressed_checkpoint_path}")
    compressed_checkpoint = torch.load(compressed_checkpoint_path, map_location='cpu')

    method = compressed_checkpoint['method']
    compressed_data = compressed_checkpoint['compressed_data']

    print(f"Decompressing with method: {method}")
    compressor = get_compressor(method)

    start_time = time.time()
    state_dict = compressor.decompress(compressed_data)
    decompression_time = time.time() - start_time

    print(f"Decompression time: {decompression_time:.2f}s")

    # 重建完整的检查点
    checkpoint = {
        'model_state_dict': state_dict,
    }

    # 恢复其他信息
    if 'optimizer_state_dict' in compressed_checkpoint:
        checkpoint['optimizer_state_dict'] = compressed_checkpoint['optimizer_state_dict']
    if 'epoch' in compressed_checkpoint:
        checkpoint['epoch'] = compressed_checkpoint['epoch']
    if 'history' in compressed_checkpoint:
        checkpoint['history'] = compressed_checkpoint['history']

    # 如果指定了输出路径，保存解压后的检查点
    if output_path:
        print(f"Saving decompressed checkpoint to: {output_path}")
        torch.save(checkpoint, output_path)

    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Compress and decompress model checkpoints'
    )

    # 操作模式
    parser.add_argument('--mode', type=str, required=True,
                        choices=['compress', 'decompress'],
                        help='Operation mode')

    # 压缩参数
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint file (for compress mode)')
    parser.add_argument('--method', type=str,
                        choices=['excp', 'inshrinkerator', 'predictive'],
                        help='Compression method (for compress mode)')
    parser.add_argument('--output', type=str,
                        help='Output file path')

    # 解压参数
    parser.add_argument('--compressed', type=str,
                        help='Path to compressed checkpoint (for decompress mode)')

    args = parser.parse_args()

    if args.mode == 'compress':
        if not args.checkpoint or not args.method or not args.output:
            parser.error("compress mode requires --checkpoint, --method, and --output")

        stats = compress_checkpoint(
            checkpoint_path=args.checkpoint,
            method=args.method,
            output_path=args.output,
        )

        print("\n" + "=" * 60)
        print("Compression Summary:")
        print("=" * 60)
        print(f"Original size: {stats['original_size'] / 1024 / 1024:.2f} MB")
        print(f"Compressed size: {stats['compressed_size'] / 1024 / 1024:.2f} MB")
        print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"Compression time: {stats['compression_time']:.2f}s")
        print(f"Space saved: {(1 - 1/stats['compression_ratio']) * 100:.1f}%")

    elif args.mode == 'decompress':
        if not args.compressed:
            parser.error("decompress mode requires --compressed")

        checkpoint = decompress_checkpoint(
            compressed_checkpoint_path=args.compressed,
            output_path=args.output,
        )

        print("\n" + "=" * 60)
        print("Decompression Summary:")
        print("=" * 60)
        print(f"Checkpoint contains {len(checkpoint['model_state_dict'])} tensors")
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")


if __name__ == '__main__':
    main()
