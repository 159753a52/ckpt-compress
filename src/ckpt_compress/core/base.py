"""
检查点压缩的基类。
"""

from abc import ABC, abstractmethod
from typing import Dict
import torch


class BaseCompressor(ABC):
    """
    所有检查点压缩方法的抽象基类。

    所有压缩方法必须继承此类并实现 compress、decompress 方法和 name 属性。
    """

    @abstractmethod
    def compress(self, state_dict: Dict[str, torch.Tensor]) -> bytes:
        """
        压缩模型状态字典。

        参数:
            state_dict: 参数名称到张量的映射字典。

        返回:
            压缩后的字节数据。
        """
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> Dict[str, torch.Tensor]:
        """
        将压缩数据解压回模型状态字典。

        参数:
            data: 压缩后的字节数据。

        返回:
            参数名称到张量的映射字典。
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        返回压缩方法的名称。

        返回:
            压缩方法的名称。
        """
        pass
