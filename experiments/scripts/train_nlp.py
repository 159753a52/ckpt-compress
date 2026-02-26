"""
NLP 模型训练脚本。

支持 GPT-2 Small/Medium 在 WikiText-2/WikiText-103 上的训练。
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import (
    get_gpt2_small,
    get_gpt2_medium,
)
from src.ckpt_compress.utils.data_loader import (
    get_wikitext2_dataloader,
    get_wikitext103_dataloader,
)
from src.ckpt_compress.utils.trainer import BaseTrainer


def get_model(model_name: str, pretrained: bool = False):
    """
    获取模型。

    参数:
        model_name: 模型名称 ('gpt2-small', 'gpt2-medium')
        pretrained: 是否使用预训练权重

    返回:
        模型实例
    """
    if model_name == 'gpt2-small':
        return get_gpt2_small(pretrained=pretrained)
    elif model_name == 'gpt2-medium':
        return get_gpt2_medium(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_data_loader(
    dataset_name: str,
    split: str,
    batch_size: int,
    seq_length: int,
    max_samples: int = None,
    num_workers: int = 0,
    local_path: str = None,
):
    """
    获取数据加载器。

    参数:
        dataset_name: 数据集名称 ('wikitext2', 'wikitext103')
        split: 数据集划分 ('train', 'validation', 'test')
        batch_size: 批次大小
        seq_length: 序列长度
        max_samples: 最大样本数
        num_workers: 工作进程数
        local_path: 本地数据路径

    返回:
        DataLoader 对象
    """
    if dataset_name == 'wikitext2':
        return get_wikitext2_dataloader(
            split=split,
            batch_size=batch_size,
            seq_length=seq_length,
            max_samples=max_samples,
            num_workers=num_workers,
            local_path=local_path,
        )
    elif dataset_name == 'wikitext103':
        return get_wikitext103_dataloader(
            split=split,
            batch_size=batch_size,
            seq_length=seq_length,
            max_samples=max_samples,
            num_workers=num_workers,
            local_path=local_path,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description='Train NLP models')

    # 模型参数
    parser.add_argument('--model', type=str, default='gpt2-small',
                        choices=['gpt2-small', 'gpt2-medium'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'wikitext103'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Local data directory (optional)')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='Sequence length')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples (for testing)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')

    # 学习率调度
    parser.add_argument('--lr_schedule', type=str, default='linear',
                        choices=['linear', 'cosine', 'none'],
                        help='Learning rate schedule')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Warmup steps for linear schedule')

    # 检查点参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # 早停参数
    parser.add_argument('--early_stopping', type=int, default=None,
                        help='Early stopping patience (None to disable)')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')

    args = parser.parse_args()

    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建模型
    print(f"Creating model: {args.model}")
    model = get_model(args.model, pretrained=args.pretrained)

    # 加载数据
    print(f"Loading dataset: {args.dataset}")
    train_loader = get_data_loader(
        args.dataset,
        split='train',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        local_path=args.data_dir,
    )

    val_loader = get_data_loader(
        args.dataset,
        split='validation',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        max_samples=args.max_samples // 10 if args.max_samples else None,
        num_workers=args.num_workers,
        local_path=args.data_dir,
    )

    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 创建学习率调度器
    scheduler = None
    if args.lr_schedule == 'linear':
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int):
            if current_step < args.warmup_steps:
                return float(current_step) / float(max(1, args.warmup_steps))
            return max(
                0.0,
                float(args.epochs * len(train_loader) - current_step) /
                float(max(1, args.epochs * len(train_loader) - args.warmup_steps))
            )

        scheduler = LambdaLR(optimizer, lr_lambda)
    elif args.lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )

    # 创建损失函数（GPT-2 模型内部已包含损失计算）
    criterion = nn.CrossEntropyLoss()

    # 创建训练器
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping,
        scheduler=scheduler,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # 如果需要，从检查点恢复
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    # 训练
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Model: {args.model}, Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print("=" * 60)

    history = trainer.train(
        epochs=args.epochs,
        save_every=args.save_every,
        verbose=True,
    )

    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()
