"""
基础压缩器类的测试。
"""

import pytest
import torch
from abc import ABC

from ckpt_compress.core.base import BaseCompressor


class TestBaseCompressor:
    """BaseCompressor 抽象基类的测试。"""

    def test_base_compressor_is_abstract(self):
        """测试 BaseCompressor 不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseCompressor()

    def test_base_compressor_requires_compress_method(self):
        """测试子类必须实现 compress 方法。"""

        class IncompleteCompressor(BaseCompressor):
            def decompress(self, data: bytes):
                return {}

            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteCompressor()

    def test_base_compressor_requires_decompress_method(self):
        """测试子类必须实现 decompress 方法。"""

        class IncompleteCompressor(BaseCompressor):
            def compress(self, state_dict):
                return b""

            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteCompressor()

    def test_base_compressor_requires_name_property(self):
        """测试子类必须实现 name 属性。"""

        class IncompleteCompressor(BaseCompressor):
            def compress(self, state_dict):
                return b""

            def decompress(self, data: bytes):
                return {}

        with pytest.raises(TypeError):
            IncompleteCompressor()

    def test_complete_subclass_can_be_instantiated(self):
        """测试完整的子类可以被实例化。"""

        class CompleteCompressor(BaseCompressor):
            def compress(self, state_dict):
                return b"compressed"

            def decompress(self, data: bytes):
                return {}

            @property
            def name(self) -> str:
                return "complete"

        compressor = CompleteCompressor()
        assert compressor is not None
        assert compressor.name == "complete"

    def test_compress_returns_bytes(self):
        """测试 compress 方法返回字节。"""

        class TestCompressor(BaseCompressor):
            def compress(self, state_dict):
                return b"test_data"

            def decompress(self, data: bytes):
                return {}

            @property
            def name(self) -> str:
                return "test"

        compressor = TestCompressor()
        result = compressor.compress({})
        assert isinstance(result, bytes)

    def test_decompress_returns_dict(self):
        """测试 decompress 方法返回字典。"""

        class TestCompressor(BaseCompressor):
            def compress(self, state_dict):
                return b""

            def decompress(self, data: bytes):
                return {"key": torch.tensor([1.0])}

            @property
            def name(self) -> str:
                return "test"

        compressor = TestCompressor()
        result = compressor.decompress(b"")
        assert isinstance(result, dict)
