"""
数据下载脚本测试。

TDD：先写测试，再实现。
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestDownloadCIFAR10:
    """CIFAR-10 数据下载测试。"""

    def test_download_cifar10_creates_directory(self, tmp_path):
        """应该创建数据目录。"""
        from scripts.download_data import download_cifar10
        
        data_dir = str(tmp_path / "data")
        with patch('torchvision.datasets.CIFAR10'):
            download_cifar10(data_dir=data_dir)
        
        assert os.path.exists(data_dir)

    def test_download_cifar10_calls_dataset(self, tmp_path):
        """应该调用 CIFAR10 数据集下载（训练集和测试集）。"""
        from scripts.download_data import download_cifar10
        
        data_dir = str(tmp_path / "data")
        with patch('torchvision.datasets.CIFAR10') as mock_cifar10:
            download_cifar10(data_dir=data_dir)
            
            # 应该调用两次：训练集和测试集
            assert mock_cifar10.call_count == 2
            
            # 检查第一次调用（训练集）
            first_call = mock_cifar10.call_args_list[0]
            assert first_call[1]['root'] == data_dir
            assert first_call[1]['train'] is True
            assert first_call[1]['download'] is True
            
            # 检查第二次调用（测试集）
            second_call = mock_cifar10.call_args_list[1]
            assert second_call[1]['root'] == data_dir
            assert second_call[1]['train'] is False
            assert second_call[1]['download'] is True

    def test_download_cifar10_returns_success(self, tmp_path):
        """成功下载应该返回 True。"""
        from scripts.download_data import download_cifar10
        
        data_dir = str(tmp_path / "data")
        with patch('torchvision.datasets.CIFAR10'):
            result = download_cifar10(data_dir=data_dir)
        
        assert result is True

    def test_download_cifar10_handles_error(self, tmp_path):
        """下载失败应该返回 False。"""
        from scripts.download_data import download_cifar10
        
        data_dir = str(tmp_path / "data")
        with patch('torchvision.datasets.CIFAR10', side_effect=Exception("Download failed")):
            result = download_cifar10(data_dir=data_dir)
        
        assert result is False


class TestDownloadCIFAR100:
    """CIFAR-100 数据下载测试。"""

    def test_download_cifar100_creates_directory(self, tmp_path):
        """应该创建数据目录。"""
        from scripts.download_data import download_cifar100
        
        data_dir = str(tmp_path / "data")
        with patch('torchvision.datasets.CIFAR100'):
            download_cifar100(data_dir=data_dir)
        
        assert os.path.exists(data_dir)

    def test_download_cifar100_calls_dataset(self, tmp_path):
        """应该调用 CIFAR100 数据集下载（训练集和测试集）。"""
        from scripts.download_data import download_cifar100
        
        data_dir = str(tmp_path / "data")
        with patch('torchvision.datasets.CIFAR100') as mock_cifar100:
            download_cifar100(data_dir=data_dir)
            
            # 应该调用两次：训练集和测试集
            assert mock_cifar100.call_count == 2
            
            # 检查第一次调用（训练集）
            first_call = mock_cifar100.call_args_list[0]
            assert first_call[1]['root'] == data_dir
            assert first_call[1]['train'] is True
            assert first_call[1]['download'] is True
            
            # 检查第二次调用（测试集）
            second_call = mock_cifar100.call_args_list[1]
            assert second_call[1]['root'] == data_dir
            assert second_call[1]['train'] is False
            assert second_call[1]['download'] is True

    def test_download_cifar100_returns_success(self, tmp_path):
        """成功下载应该返回 True。"""
        from scripts.download_data import download_cifar100
        
        data_dir = str(tmp_path / "data")
        with patch('torchvision.datasets.CIFAR100'):
            result = download_cifar100(data_dir=data_dir)
        
        assert result is True


class TestDownloadWikiText2:
    """WikiText-2 数据下载测试。"""

    def test_download_wikitext2_creates_directory(self, tmp_path):
        """应该创建数据目录。"""
        from scripts.download_data import download_wikitext2
        
        data_dir = str(tmp_path / "data")
        with patch('datasets.load_dataset'):
            download_wikitext2(data_dir=data_dir)
        
        assert os.path.exists(data_dir)

    def test_download_wikitext2_calls_load_dataset(self, tmp_path):
        """应该调用 load_dataset。"""
        from scripts.download_data import download_wikitext2
        
        data_dir = str(tmp_path / "data")
        with patch('datasets.load_dataset') as mock_load:
            download_wikitext2(data_dir=data_dir)
            
            mock_load.assert_called_once_with(
                'wikitext', 'wikitext-2-raw-v1', cache_dir=data_dir
            )

    def test_download_wikitext2_returns_success(self, tmp_path):
        """成功下载应该返回 True。"""
        from scripts.download_data import download_wikitext2
        
        data_dir = str(tmp_path / "data")
        with patch('datasets.load_dataset'):
            result = download_wikitext2(data_dir=data_dir)
        
        assert result is True


class TestDownloadWikiText103:
    """WikiText-103 数据下载测试。"""

    def test_download_wikitext103_creates_directory(self, tmp_path):
        """应该创建数据目录。"""
        from scripts.download_data import download_wikitext103
        
        data_dir = str(tmp_path / "data")
        with patch('datasets.load_dataset'):
            download_wikitext103(data_dir=data_dir)
        
        assert os.path.exists(data_dir)

    def test_download_wikitext103_calls_load_dataset(self, tmp_path):
        """应该调用 load_dataset。"""
        from scripts.download_data import download_wikitext103
        
        data_dir = str(tmp_path / "data")
        with patch('datasets.load_dataset') as mock_load:
            download_wikitext103(data_dir=data_dir)
            
            mock_load.assert_called_once_with(
                'wikitext', 'wikitext-103-raw-v1', cache_dir=data_dir
            )

    def test_download_wikitext103_returns_success(self, tmp_path):
        """成功下载应该返回 True。"""
        from scripts.download_data import download_wikitext103
        
        data_dir = str(tmp_path / "data")
        with patch('datasets.load_dataset'):
            result = download_wikitext103(data_dir=data_dir)
        
        assert result is True


class TestDownloadAll:
    """下载所有数据集测试。"""

    def test_download_all_default(self, tmp_path):
        """默认应该下载所有数据集。"""
        from scripts.download_data import download_all
        
        data_dir = str(tmp_path / "data")
        with patch('scripts.download_data.download_cifar10') as mock_c10, \
             patch('scripts.download_data.download_cifar100') as mock_c100, \
             patch('scripts.download_data.download_wikitext2') as mock_wt2, \
             patch('scripts.download_data.download_wikitext103') as mock_wt103:
            
            download_all(data_dir=data_dir)
            
            mock_c10.assert_called_once_with(data_dir)
            mock_c100.assert_called_once_with(data_dir)
            mock_wt2.assert_called_once_with(data_dir)
            mock_wt103.assert_called_once_with(data_dir)

    def test_download_all_specific_datasets(self, tmp_path):
        """应该只下载指定的数据集。"""
        from scripts.download_data import download_all
        
        data_dir = str(tmp_path / "data")
        datasets = ['cifar10', 'wikitext2']
        
        with patch('scripts.download_data.download_cifar10') as mock_c10, \
             patch('scripts.download_data.download_cifar100') as mock_c100, \
             patch('scripts.download_data.download_wikitext2') as mock_wt2, \
             patch('scripts.download_data.download_wikitext103') as mock_wt103:
            
            download_all(data_dir=data_dir, datasets=datasets)
            
            mock_c10.assert_called_once_with(data_dir)
            mock_c100.assert_not_called()
            mock_wt2.assert_called_once_with(data_dir)
            mock_wt103.assert_not_called()

    def test_download_all_returns_results(self, tmp_path):
        """应该返回下载结果字典。"""
        from scripts.download_data import download_all
        
        data_dir = str(tmp_path / "data")
        with patch('scripts.download_data.download_cifar10', return_value=True), \
             patch('scripts.download_data.download_cifar100', return_value=True), \
             patch('scripts.download_data.download_wikitext2', return_value=False), \
             patch('scripts.download_data.download_wikitext103', return_value=True):
            
            results = download_all(data_dir=data_dir)
            
            assert results['cifar10'] is True
            assert results['cifar100'] is True
            assert results['wikitext2'] is False
            assert results['wikitext103'] is True


class TestVerifyDataset:
    """数据集验证测试。"""

    def test_verify_cifar10_exists(self, tmp_path):
        """CIFAR-10 存在时应该返回 True。"""
        from scripts.download_data import verify_dataset
        
        data_dir = tmp_path / "data"
        cifar_dir = data_dir / "cifar-10-batches-py"
        cifar_dir.mkdir(parents=True)
        
        result = verify_dataset('cifar10', str(data_dir))
        assert result is True

    def test_verify_cifar10_not_exists(self, tmp_path):
        """CIFAR-10 不存在时应该返回 False。"""
        from scripts.download_data import verify_dataset
        
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        
        result = verify_dataset('cifar10', str(data_dir))
        assert result is False

    def test_verify_cifar100_exists(self, tmp_path):
        """CIFAR-100 存在时应该返回 True。"""
        from scripts.download_data import verify_dataset
        
        data_dir = tmp_path / "data"
        cifar_dir = data_dir / "cifar-100-python"
        cifar_dir.mkdir(parents=True)
        
        result = verify_dataset('cifar100', str(data_dir))
        assert result is True

    def test_verify_invalid_dataset(self, tmp_path):
        """无效数据集名称应该抛出错误。"""
        from scripts.download_data import verify_dataset
        
        data_dir = str(tmp_path / "data")
        with pytest.raises(ValueError, match="Unknown dataset"):
            verify_dataset('invalid_dataset', data_dir)

    def test_verify_wikitext2_exists(self, tmp_path):
        """WikiText-2 存在时应该返回 True。"""
        from scripts.download_data import verify_dataset
        
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        
        result = verify_dataset('wikitext2', str(data_dir))
        assert result is True

    def test_verify_wikitext103_exists(self, tmp_path):
        """WikiText-103 存在时应该返回 True。"""
        from scripts.download_data import verify_dataset
        
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        
        result = verify_dataset('wikitext103', str(data_dir))
        assert result is True


class TestDownloadErrors:
    """下载错误处理测试。"""

    def test_download_cifar100_handles_error(self, tmp_path):
        """CIFAR-100 下载失败应该返回 False。"""
        from scripts.download_data import download_cifar100
        
        data_dir = str(tmp_path / "data")
        with patch('torchvision.datasets.CIFAR100', side_effect=Exception("Download failed")):
            result = download_cifar100(data_dir=data_dir)
        
        assert result is False

    def test_download_wikitext2_handles_error(self, tmp_path):
        """WikiText-2 下载失败应该返回 False。"""
        from scripts.download_data import download_wikitext2
        
        data_dir = str(tmp_path / "data")
        with patch('datasets.load_dataset', side_effect=Exception("Download failed")):
            result = download_wikitext2(data_dir=data_dir)
        
        assert result is False

    def test_download_wikitext103_handles_error(self, tmp_path):
        """WikiText-103 下载失败应该返回 False。"""
        from scripts.download_data import download_wikitext103
        
        data_dir = str(tmp_path / "data")
        with patch('datasets.load_dataset', side_effect=Exception("Download failed")):
            result = download_wikitext103(data_dir=data_dir)
        
        assert result is False


class TestDownloadAllErrors:
    """下载所有数据集错误处理测试。"""

    def test_download_all_with_unknown_dataset(self, tmp_path):
        """未知数据集应该返回 False。"""
        from scripts.download_data import download_all
        
        data_dir = str(tmp_path / "data")
        datasets = ['unknown_dataset']
        
        results = download_all(data_dir=data_dir, datasets=datasets)
        
        assert results['unknown_dataset'] is False

    def test_download_all_mixed_results(self, tmp_path):
        """混合成功和失败的结果。"""
        from scripts.download_data import download_all
        
        data_dir = str(tmp_path / "data")
        datasets = ['cifar10', 'unknown']
        
        with patch('scripts.download_data.download_cifar10', return_value=True):
            results = download_all(data_dir=data_dir, datasets=datasets)
        
        assert results['cifar10'] is True
        assert results['unknown'] is False


class TestMainCLI:
    """命令行接口测试。"""

    def test_main_with_all_flag(self, tmp_path, monkeypatch):
        """--all 标志应该下载所有数据集。"""
        import sys
        from scripts.download_data import main
        
        data_dir = str(tmp_path / "data")
        test_args = ['download_data.py', '--all', '--data_dir', data_dir]
        
        with patch('scripts.download_data.download_all') as mock_download_all:
            monkeypatch.setattr(sys, 'argv', test_args)
            main()
            
            mock_download_all.assert_called_once_with(data_dir=data_dir)

    def test_main_with_dataset_flag(self, tmp_path, monkeypatch):
        """--dataset 标志应该下载指定数据集。"""
        import sys
        from scripts.download_data import main
        
        data_dir = str(tmp_path / "data")
        test_args = ['download_data.py', '--dataset', 'cifar10,wikitext2', '--data_dir', data_dir]
        
        with patch('scripts.download_data.download_all') as mock_download_all:
            monkeypatch.setattr(sys, 'argv', test_args)
            main()
            
            mock_download_all.assert_called_once()
            call_kwargs = mock_download_all.call_args[1]
            assert call_kwargs['data_dir'] == data_dir
            assert call_kwargs['datasets'] == ['cifar10', 'wikitext2']

    def test_main_with_verify_flag_exists(self, tmp_path, monkeypatch, capsys):
        """--verify 标志应该验证数据集存在。"""
        import sys
        from scripts.download_data import main
        
        data_dir = tmp_path / "data"
        cifar_dir = data_dir / "cifar-10-batches-py"
        cifar_dir.mkdir(parents=True)
        
        test_args = ['download_data.py', '--verify', 'cifar10', '--data_dir', str(data_dir)]
        
        monkeypatch.setattr(sys, 'argv', test_args)
        main()
        
        captured = capsys.readouterr()
        assert 'cifar10 exists' in captured.out

    def test_main_with_verify_flag_not_exists(self, tmp_path, monkeypatch, capsys):
        """--verify 标志应该检测数据集不存在。"""
        import sys
        from scripts.download_data import main
        
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        
        test_args = ['download_data.py', '--verify', 'cifar10', '--data_dir', str(data_dir)]
        
        monkeypatch.setattr(sys, 'argv', test_args)
        main()
        
        captured = capsys.readouterr()
        assert 'not found' in captured.out

    def test_main_with_default_data_dir(self, monkeypatch):
        """默认数据目录应该是 ./data。"""
        import sys
        from scripts.download_data import main
        
        test_args = ['download_data.py', '--dataset', 'cifar10']
        
        with patch('scripts.download_data.download_all') as mock_download_all:
            monkeypatch.setattr(sys, 'argv', test_args)
            main()
            
            call_kwargs = mock_download_all.call_args[1]
            assert call_kwargs['data_dir'] == './data'
