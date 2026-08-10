"""
INT4 量化器

将权重量化为 4-bit 整数（16 个量化级别）。
"""

import math

import torch
from typing import Tuple, Dict, Any


class INT4Quantizer:
    """
    INT4 量化器
    
    使用线性量化将权重映射到 [-8, 7] 的 4-bit 整数范围。
    
    使用示例:
        quantizer = INT4Quantizer()
        quantized, metadata = quantizer.quantize(weight_tensor)
        recovered = quantizer.dequantize(quantized, metadata)
    """
    
    def __init__(self, quant_range: int = 8):
        """
        初始化 INT4 量化器
        
        Args:
            quant_range: 量化范围（对称量化，范围为 [-range, range-1]）
        """
        if not isinstance(quant_range, int) or isinstance(quant_range, bool) or quant_range < 2:
            raise ValueError(f"quant_range must be an integer >= 2, got {quant_range}")
        self.quant_range = quant_range
        self.n_levels = 2 * quant_range  # 16 levels for 4-bit
    
    def quantize(
        self,
        weight: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        对权重进行 INT4 量化
        
        Args:
            weight: 输入权重张量
            
        Returns:
            quantized: 量化后的 INT4 张量（存储为 int8）
            metadata: 解量化所需的元数据
                - scale: 量化缩放因子
                - zero_point: 零点偏移
                - shape: 原始形状
                - dtype: 原始数据类型
        """
        if weight.numel() == 0:
            raise ValueError("cannot quantize an empty tensor")

        # 保存原始形状和数据类型
        shape = weight.shape
        dtype = weight.dtype
        
        # 计算量化参数（对称量化）
        weight_flat = weight.detach().flatten()
        max_abs = torch.max(torch.abs(weight_flat))
        
        # 对称量化：scale = max_abs / (quant_range - 1)，映射 [-max_abs, max_abs] → [-7, 7]
        scale = max_abs / (self.quant_range - 1)
        scale = torch.clamp(scale, min=1e-8)
        
        # 量化：将权重映射到 [-8, 7]
        quantized = torch.round(weight / scale)
        quantized = torch.clamp(quantized, -self.quant_range, self.quant_range - 1)
        
        # 存储为 int8（实际只使用 4-bit）
        quantized = quantized.to(torch.int8)
        
        # 构建元数据
        metadata = {
            'scale': scale,
            'zero_point': 0,  # 对称量化不需要零点偏移
            'shape': shape,
            'dtype': dtype,
            'quant_range': self.quant_range
        }
        
        return quantized, metadata
    
    def dequantize(
        self,
        quantized: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """
        从 INT4 量化恢复权重
        
        Args:
            quantized: 量化张量（int8）
            metadata: 量化元数据
            
        Returns:
            weight: 恢复的权重张量
        """
        # 获取量化参数
        scale = metadata['scale']
        shape = metadata['shape']
        dtype = metadata['dtype']
        
        # 解量化：乘以缩放因子
        weight = quantized.float() * scale
        
        # 恢复形状和数据类型
        weight = weight.view(shape).to(dtype)
        
        return weight
    
    def compute_quantization_error(
        self,
        original: torch.Tensor,
        recovered: torch.Tensor
    ) -> float:
        """
        计算量化误差（相对 MSE）
        
        Args:
            original: 原始权重
            recovered: 量化恢复的权重
            
        Returns:
            relative_mse: 相对均方误差
        """
        mse = torch.mean((original - recovered) ** 2)
        original_var = torch.var(original)
        relative_mse = (mse / (original_var + 1e-8)).item()
        return relative_mse
    
    def get_compression_ratio(self) -> float:
        """
        计算压缩比
        
        Returns:
            compression_ratio: 压缩比（原始大小 / 压缩后大小）
            
        说明:
            - 原始权重：32-bit float
            - 量化后：4-bit int
            - 理论压缩比：32 / 4 = 8×
            - 实际压缩比略低（需要存储 scale 等元数据）
        """
        bits_per_value = math.ceil(math.log2(self.n_levels))
        return 32.0 / bits_per_value
