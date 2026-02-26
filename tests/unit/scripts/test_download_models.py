"""
模型权重下载脚本测试。

TDD：先写测试，再实现。
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestDownloadGPT2Small:
    """GPT-2 Small 模型下载测试。"""

    def test_download_gpt2_small_creates_directory(self, tmp_path):
        """应该创建缓存目录。"""
        from scripts.download_models import download_gpt2_small
        
        cache_dir = str(tmp_path / "models")
        with patch('transformers.GPT2LMHeadModel.from_pretrained'):
            download_gpt2_small(cache_dir=cache_dir)
        
        assert os.path.exists(cache_dir)

    def test_download_gpt2_small_calls_from_pretrained(self, tmp_path):
        """应该调用 from_pretrained 下载模型。"""
        from scripts.download_models import download_gpt2_small
        
        cache_dir = str(tmp_path / "models")
        with patch('transformers.GPT2LMHeadModel.from_pretrained') as mock_model:
            download_gpt2_small(cache_dir=cache_dir)
            
            mock_model.assert_called_once_with('gpt2', cache_dir=cache_dir)

    def test_download_gpt2_small_returns_success(self, tmp_path):
        """成功下载应该返回 True。"""
        from scripts.download_models import download_gpt2_small
        
        cache_dir = str(tmp_path / "models")
        with patch('transformers.GPT2LMHeadModel.from_pretrained'):
            result = download_gpt2_small(cache_dir=cache_dir)
        
        assert result is True

    def test_download_gpt2_small_handles_error(self, tmp_path):
        """下载失败应该返回 False。"""
        from scripts.download_models import download_gpt2_small
        
        cache_dir = str(tmp_path / "models")
        with patch('transformers.GPT2LMHeadModel.from_pretrained', side_effect=Exception("Download failed")):
            result = download_gpt2_small(cache_dir=cache_dir)
        
        assert result is False


class TestDownloadGPT2Medium:
    """GPT-2 Medium 模型下载测试。"""

    def test_download_gpt2_medium_creates_directory(self, tmp_path):
        """应该创建缓存目录。"""
        from scripts.download_models import download_gpt2_medium
        
        cache_dir = str(tmp_path / "models")
        with patch('transformers.GPT2LMHeadModel.from_pretrained'):
            download_gpt2_medium(cache_dir=cache_dir)
        
        assert os.path.exists(cache_dir)

    def test_download_gpt2_medium_calls_from_pretrained(self, tmp_path):
        """应该调用 from_pretrained 下载模型。"""
        from scripts.download_models import download_gpt2_medium
        
        cache_dir = str(tmp_path / "models")
        with patch('transformers.GPT2LMHeadModel.from_pretrained') as mock_model:
            download_gpt2_medium(cache_dir=cache_dir)
            
            mock_model.assert_called_once_with('gpt2-medium', cache_dir=cache_dir)

    def test_download_gpt2_medium_returns_success(self, tmp_path):
        """成功下载应该返回 True。"""
        from scripts.download_models import download_gpt2_medium
        
        cache_dir = str(tmp_path / "models")
        with patch('transformers.GPT2LMHeadModel.from_pretrained'):
            result = download_gpt2_medium(cache_dir=cache_dir)
        
        assert result is True

    def test_download_gpt2_medium_handles_error(self, tmp_path):
        """下载失败应该返回 False。"""
        from scripts.download_models import download_gpt2_medium
        
        cache_dir = str(tmp_path / "models")
        with patch('transformers.GPT2LMHeadModel.from_pretrained', side_effect=Exception("Download failed")):
            result = download_gpt2_medium(cache_dir=cache_dir)
        
        assert result is False


class TestDownloadResNet50:
    """ResNet-50 模型下载测试。"""

    def test_download_resnet50_creates_directory(self, tmp_path):
        """应该创建缓存目录。"""
        from scripts.download_models import download_resnet50_imagenet
        
        cache_dir = str(tmp_path / "models")
        with patch('torchvision.models.resnet50'):
            download_resnet50_imagenet(cache_dir=cache_dir)
        
        assert os.path.exists(cache_dir)

    def test_download_resnet50_calls_model(self, tmp_path):
        """应该调用 resnet50 下载模型。"""
        from scripts.download_models import download_resnet50_imagenet
        
        cache_dir = str(tmp_path / "models")
        with patch('torchvision.models.resnet50') as mock_model:
            download_resnet50_imagenet(cache_dir=cache_dir)
            
            mock_model.assert_called_once_with(pretrained=True)

    def test_download_resnet50_returns_success(self, tmp_path):
        """成功下载应该返回 True。"""
        from scripts.download_models import download_resnet50_imagenet
        
        cache_dir = str(tmp_path / "models")
        with patch('torchvision.models.resnet50'):
            result = download_resnet50_imagenet(cache_dir=cache_dir)
        
        assert result is True

    def test_download_resnet50_handles_error(self, tmp_path):
        """下载失败应该返回 False。"""
        from scripts.download_models import download_resnet50_imagenet
        
        cache_dir = str(tmp_path / "models")
        with patch('torchvision.models.resnet50', side_effect=Exception("Download failed")):
            result = download_resnet50_imagenet(cache_dir=cache_dir)
        
        assert result is False


class TestDownloadAllModels:
    """下载所有模型测试。"""

    def test_download_all_models_default(self, tmp_path):
        """默认应该下载所有模型。"""
        from scripts.download_models import download_all_models
        
        cache_dir = str(tmp_path / "models")
        with patch('scripts.download_models.download_gpt2_small') as mock_gpt2s, \
             patch('scripts.download_models.download_gpt2_medium') as mock_gpt2m, \
             patch('scripts.download_models.download_resnet50_imagenet') as mock_resnet:
            
            download_all_models(cache_dir=cache_dir)
            
            mock_gpt2s.assert_called_once_with(cache_dir)
            mock_gpt2m.assert_called_once_with(cache_dir)
            mock_resnet.assert_called_once_with(cache_dir)

    def test_download_all_models_specific(self, tmp_path):
        """应该只下载指定的模型。"""
        from scripts.download_models import download_all_models
        
        cache_dir = str(tmp_path / "models")
        models = ['gpt2-small', 'resnet50']
        
        with patch('scripts.download_models.download_gpt2_small') as mock_gpt2s, \
             patch('scripts.download_models.download_gpt2_medium') as mock_gpt2m, \
             patch('scripts.download_models.download_resnet50_imagenet') as mock_resnet:
            
            download_all_models(cache_dir=cache_dir, models=models)
            
            mock_gpt2s.assert_called_once_with(cache_dir)
            mock_gpt2m.assert_not_called()
            mock_resnet.assert_called_once_with(cache_dir)

    def test_download_all_models_returns_results(self, tmp_path):
        """应该返回下载结果字典。"""
        from scripts.download_models import download_all_models
        
        cache_dir = str(tmp_path / "models")
        with patch('scripts.download_models.download_gpt2_small', return_value=True), \
             patch('scripts.download_models.download_gpt2_medium', return_value=False), \
             patch('scripts.download_models.download_resnet50_imagenet', return_value=True):
            
            results = download_all_models(cache_dir=cache_dir)
            
            assert results['gpt2-small'] is True
            assert results['gpt2-medium'] is False
            assert results['resnet50'] is True


class TestDownloadAllModelsErrors:
    """下载所有模型错误处理测试。"""

    def test_download_all_models_with_unknown_model(self, tmp_path):
        """未知模型应该返回 False。"""
        from scripts.download_models import download_all_models
        
        cache_dir = str(tmp_path / "models")
        models = ['unknown_model']
        
        results = download_all_models(cache_dir=cache_dir, models=models)
        
        assert results['unknown_model'] is False

    def test_download_all_models_mixed_results(self, tmp_path):
        """混合成功和失败的结果。"""
        from scripts.download_models import download_all_models
        
        cache_dir = str(tmp_path / "models")
        models = ['gpt2-small', 'unknown']
        
        with patch('scripts.download_models.download_gpt2_small', return_value=True):
            results = download_all_models(cache_dir=cache_dir, models=models)
        
        assert results['gpt2-small'] is True
        assert results['unknown'] is False


class TestMainCLI:
    """命令行接口测试。"""

    def test_main_with_all_flag(self, tmp_path, monkeypatch):
        """--all 标志应该下载所有模型。"""
        import sys
        from scripts.download_models import main
        
        cache_dir = str(tmp_path / "models")
        test_args = ['download_models.py', '--all', '--cache_dir', cache_dir]
        
        with patch('scripts.download_models.download_all_models') as mock_download:
            monkeypatch.setattr(sys, 'argv', test_args)
            main()
            
            mock_download.assert_called_once_with(cache_dir=cache_dir, models=None)

    def test_main_with_model_flag(self, tmp_path, monkeypatch):
        """--model 标志应该下载指定模型。"""
        import sys
        from scripts.download_models import main
        
        cache_dir = str(tmp_path / "models")
        test_args = ['download_models.py', '--model', 'gpt2-small', '--cache_dir', cache_dir]
        
        with patch('scripts.download_models.download_all_models') as mock_download:
            monkeypatch.setattr(sys, 'argv', test_args)
            main()
            
            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs['cache_dir'] == cache_dir
            assert call_kwargs['models'] == ['gpt2-small']

    def test_main_with_default_cache_dir(self, monkeypatch):
        """默认缓存目录应该是 ./data/models。"""
        import sys
        from scripts.download_models import main
        
        test_args = ['download_models.py', '--model', 'gpt2-small']
        
        with patch('scripts.download_models.download_all_models') as mock_download:
            monkeypatch.setattr(sys, 'argv', test_args)
            main()
            
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs['cache_dir'] == './data/models'
