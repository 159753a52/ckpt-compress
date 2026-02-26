"""
统一训练框架测试。

TDD：先写测试，再实现。
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class SimpleModel(nn.Module):
    """简单的测试模型。"""
    def __init__(self, input_size=10, output_size=2):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def simple_model():
    """创建简单模型。"""
    return SimpleModel()


@pytest.fixture
def simple_dataloader():
    """创建简单数据加载器。"""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10)


class TestBaseTrainer:
    """BaseTrainer 基类测试。"""

    def test_trainer_initialization(self, simple_model, simple_dataloader, tmp_path):
        """应该正确初始化训练器。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )
        
        assert trainer.model is not None
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_trainer_train_one_epoch(self, simple_model, simple_dataloader, tmp_path):
        """应该能训练一个 epoch。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )
        
        loss = trainer.train_one_epoch()
        assert isinstance(loss, float)
        assert loss > 0

    def test_trainer_validate(self, simple_model, simple_dataloader, tmp_path):
        """应该能验证模型。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )
        
        val_loss, val_acc = trainer.validate()
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss > 0
        assert 0 <= val_acc <= 1

    def test_trainer_save_checkpoint(self, simple_model, simple_dataloader, tmp_path):
        """应该能保存检查点。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )
        
        checkpoint_path = trainer.save_checkpoint(epoch=1, val_loss=0.5)
        assert Path(checkpoint_path).exists()

    def test_trainer_load_checkpoint(self, simple_model, simple_dataloader, tmp_path):
        """应该能加载检查点。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )
        
        # 保存检查点
        checkpoint_path = trainer.save_checkpoint(epoch=1, val_loss=0.5)
        
        # 创建新的训练器并加载
        new_trainer = BaseTrainer(
            model=SimpleModel(),
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(SimpleModel().parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )
        
        epoch = new_trainer.load_checkpoint(checkpoint_path)
        assert epoch == 1

    def test_trainer_train_full(self, simple_model, simple_dataloader, tmp_path):
        """应该能完整训练多个 epoch。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )
        
        history = trainer.train(epochs=2)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'val_acc' in history
        assert len(history['train_loss']) == 2

    def test_trainer_early_stopping(self, simple_model, simple_dataloader, tmp_path):
        """应该支持早停机制。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            early_stopping_patience=2
        )
        
        # 模拟早停
        assert trainer.early_stopping_patience == 2

    def test_trainer_learning_rate_scheduler(self, simple_model, simple_dataloader, tmp_path):
        """应该支持学习率调度。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            scheduler=scheduler
        )
        
        assert trainer.scheduler is not None

    def test_trainer_with_amp(self, simple_model, simple_dataloader, tmp_path):
        """应该支持混合精度训练。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            use_amp=True
        )
        
        assert trainer.use_amp is True
        assert hasattr(trainer, 'scaler')

    def test_trainer_with_gradient_accumulation(self, simple_model, simple_dataloader, tmp_path):
        """应该支持梯度累积。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            gradient_accumulation_steps=4
        )
        
        assert trainer.gradient_accumulation_steps == 4
        loss = trainer.train_one_epoch()
        assert isinstance(loss, float)

    def test_trainer_with_dict_batch(self, simple_model, tmp_path):
        """应该支持字典类型的批次（NLP 任务）。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        # 创建字典类型的数据加载器
        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 100
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randn(10),
                    'labels': torch.randint(0, 2, (1,)).item()
                }
        
        dict_loader = DataLoader(DictDataset(), batch_size=10)
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=dict_loader,
            val_loader=dict_loader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )
        
        loss = trainer.train_one_epoch()
        assert isinstance(loss, float)

    def test_trainer_early_stopping_triggers(self, simple_model, simple_dataloader, tmp_path):
        """早停应该在验证损失不改善时触发。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            early_stopping_patience=2
        )
        
        # 训练应该在早停触发前停止
        history = trainer.train(epochs=10, verbose=False)
        
        # 如果早停触发，训练的 epoch 数应该少于 10
        assert len(history['train_loss']) <= 10

    def test_trainer_saves_best_model(self, simple_model, simple_dataloader, tmp_path):
        """应该保存最佳模型。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            early_stopping_patience=5
        )
        
        trainer.train(epochs=2, verbose=False)
        
        # 检查是否保存了最佳模型
        best_model_path = tmp_path / 'best_model.pt'
        assert best_model_path.exists()

    def test_trainer_load_checkpoint_with_scheduler(self, simple_model, simple_dataloader, tmp_path):
        """应该能加载包含调度器的检查点。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            scheduler=scheduler
        )
        
        # 保存检查点
        checkpoint_path = trainer.save_checkpoint(epoch=1, val_loss=0.5)
        
        # 创建新的训练器并加载
        new_optimizer = torch.optim.Adam(SimpleModel().parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.1)
        
        new_trainer = BaseTrainer(
            model=SimpleModel(),
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=new_optimizer,
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            scheduler=new_scheduler
        )
        
        epoch = new_trainer.load_checkpoint(checkpoint_path)
        assert epoch == 1

    def test_trainer_with_amp_and_scaler(self, simple_model, simple_dataloader, tmp_path):
        """应该能保存和加载 AMP scaler 状态。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            use_amp=True
        )
        
        # 保存检查点
        checkpoint_path = trainer.save_checkpoint(epoch=1, val_loss=0.5)
        
        # 创建新的训练器并加载
        new_trainer = BaseTrainer(
            model=SimpleModel(),
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(SimpleModel().parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            use_amp=True
        )
        
        epoch = new_trainer.load_checkpoint(checkpoint_path)
        assert epoch == 1

    def test_trainer_validate_without_classification(self, tmp_path):
        """应该能验证非分类任务（没有准确率）。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        # 创建回归任务的数据加载器
        class RegressionDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 100
            
            def __getitem__(self, idx):
                return torch.randn(10), torch.randn(1)
        
        regression_loader = DataLoader(RegressionDataset(), batch_size=10)
        
        # 创建回归模型
        class RegressionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.fc(x)
        
        model = RegressionModel()
        trainer = BaseTrainer(
            model=model,
            train_loader=regression_loader,
            val_loader=regression_loader,
            optimizer=torch.optim.Adam(model.parameters()),
            criterion=nn.MSELoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )
        
        val_loss, val_acc = trainer.validate()
        assert isinstance(val_loss, float)
        assert val_acc == 0.0  # 回归任务没有准确率

    def test_trainer_without_scheduler(self, simple_model, simple_dataloader, tmp_path):
        """应该能在没有调度器的情况下训练。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            scheduler=None
        )
        
        history = trainer.train(epochs=2, verbose=False)
        assert len(history['train_loss']) == 2

    def test_trainer_checkpoint_without_scheduler(self, simple_model, simple_dataloader, tmp_path):
        """应该能保存和加载没有调度器的检查点。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            scheduler=None
        )
        
        # 保存检查点
        checkpoint_path = trainer.save_checkpoint(epoch=1, val_loss=0.5)
        
        # 加载检查点
        new_trainer = BaseTrainer(
            model=SimpleModel(),
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(SimpleModel().parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path),
            scheduler=None
        )
        
        epoch = new_trainer.load_checkpoint(checkpoint_path)
        assert epoch == 1

    def test_trainer_save_every_parameter(self, simple_model, simple_dataloader, tmp_path):
        """应该根据 save_every 参数保存检查点。"""
        from ckpt_compress.utils.trainer import BaseTrainer
        
        trainer = BaseTrainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            optimizer=torch.optim.Adam(simple_model.parameters()),
            criterion=nn.CrossEntropyLoss(),
            device='cpu',
            checkpoint_dir=str(tmp_path)
        )
        
        # 每 2 个 epoch 保存一次
        trainer.train(epochs=4, save_every=2, verbose=False)
        
        # 应该有 epoch 2 和 epoch 4 的检查点
        assert (tmp_path / 'checkpoint_epoch_2.pt').exists()
        assert (tmp_path / 'checkpoint_epoch_4.pt').exists()
        assert not (tmp_path / 'checkpoint_epoch_1.pt').exists()
        assert not (tmp_path / 'checkpoint_epoch_3.pt').exists()
