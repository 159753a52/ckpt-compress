"""
统一训练框架。

提供 BaseTrainer 基类和 CV/NLP 任务的训练器。
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
from pathlib import Path


def _validate_positive_int(name: str, value: int) -> int:
    """Validate an integer training-loop control parameter."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


class BaseTrainer:
    """
    统一训练框架基类。
    
    功能:
    - 训练循环
    - 验证循环
    - 检查点保存/加载
    - 日志记录
    - 早停
    - 学习率调度
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        early_stopping_patience: Optional[int] = None,
        scheduler: Optional[Any] = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
    ):
        """
        初始化训练器。
        
        参数:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            criterion: 损失函数
            device: 设备 ('cuda' 或 'cpu')
            checkpoint_dir: 检查点保存目录
            early_stopping_patience: 早停耐心值（None 表示不使用早停）
            scheduler: 学习率调度器
            use_amp: 是否使用自动混合精度训练
            gradient_accumulation_steps: 梯度累积步数
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.early_stopping_patience = early_stopping_patience
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.gradient_accumulation_steps = _validate_positive_int(
            "gradient_accumulation_steps", gradient_accumulation_steps
        )
        
        # 早停相关
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 混合精度训练
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
        }
    
    def _prepare_batch(self, batch):
        """Move a tuple, mapping, or tensor batch to the trainer device."""
        if isinstance(batch, (tuple, list)):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)
            return inputs, targets

        if isinstance(batch, dict):
            if 'input_ids' in batch:
                inputs = batch['input_ids'].to(self.device)
            else:
                inputs = next(
                    value for value in batch.values()
                    if isinstance(value, torch.Tensor)
                ).to(self.device)
            targets = batch.get('labels')
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)
            return inputs, targets

        return batch.to(self.device), None

    def _compute_loss(self, inputs, targets):
        """Run the model and compute classification or shifted LM loss."""
        outputs = self.model(inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        if targets is None:
            return outputs.loss if hasattr(outputs, 'loss') else outputs, logits
        if logits.dim() == 3:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        else:
            loss = self.criterion(logits, targets)
        return loss, logits

    def _step_optimizer(self):
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

    def train_one_epoch(self) -> float:
        """
        训练一个 epoch。
        
        返回:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs, targets = self._prepare_batch(batch)
            
            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss, _ = self._compute_loss(inputs, targets)
            else:
                loss, _ = self._compute_loss(inputs, targets)
            
            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 更新参数
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self._step_optimizer()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

        # Do not silently discard gradients from a final partial accumulation.
        if num_batches and num_batches % self.gradient_accumulation_steps:
            self._step_optimizer()
        
        if num_batches == 0:
            raise ValueError("train_loader must contain at least one batch")
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> tuple:
        """
        验证模型。
        
        返回:
            tuple: (验证损失, 验证准确率)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                num_batches += 1
                inputs, targets = self._prepare_batch(batch)
                loss, logits = self._compute_loss(inputs, targets)
                
                total_loss += loss.item()
                
                # 计算准确率（仅分类任务）
                if targets is not None and logits.dim() == 2:
                    _, predicted = torch.max(logits.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
        
        if num_batches == 0:
            raise ValueError("val_loader must contain at least one batch")
        avg_loss = total_loss / num_batches
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        filename: Optional[str] = None
    ) -> str:
        """
        保存检查点。
        
        参数:
            epoch: 当前 epoch
            val_loss: 验证损失
            filename: 检查点文件名（可选）
            
        返回:
            str: 检查点文件路径
        """
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pt'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        加载检查点。
        
        参数:
            checkpoint_path: 检查点文件路径
            
        返回:
            int: 加载的 epoch
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        return checkpoint['epoch']
    
    def train(
        self,
        epochs: int,
        save_every: int = 1,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        完整训练循环。
        
        参数:
            epochs: 训练轮数
            save_every: 每隔多少个 epoch 保存一次检查点
            verbose: 是否打印训练信息
            
        返回:
            Dict[str, List[float]]: 训练历史
        """
        _validate_positive_int("epochs", epochs)
        _validate_positive_int("save_every", save_every)
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # 训练一个 epoch
            train_loss = self.train_one_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 打印信息
            if verbose:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch}/{epochs} - "
                      f"train_loss: {train_loss:.4f} - "
                      f"val_loss: {val_loss:.4f} - "
                      f"val_acc: {val_acc:.4f} - "
                      f"time: {epoch_time:.2f}s")
            
            # 保存检查点
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, val_loss)
            
            # 早停检查
            if self.early_stopping_patience is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # 保存最佳模型
                    self.save_checkpoint(epoch, val_loss, filename='best_model.pt')
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch}")
                        break
        
        return self.history
