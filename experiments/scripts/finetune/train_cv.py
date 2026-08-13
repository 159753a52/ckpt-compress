"""
CV 模型训练脚本。

支持 ResNet18/ResNet50 在 CIFAR-10/CIFAR-100 上的训练。
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dacp.models.resnet import get_resnet18_cifar10, get_resnet50_cifar100
from dacp.utils.data_loader import get_cifar10_loaders, get_cifar100_loaders
from dacp.utils.trainer import BaseTrainer


def get_model(model_name: str, num_classes: int):
    """
    获取模型。

    参数:
        model_name: 模型名称 ('resnet18', 'resnet50')
        num_classes: 类别数

    返回:
        模型实例
    """
    if model_name == "resnet18":
        return get_resnet18_cifar10(num_classes=num_classes)
    elif model_name == "resnet50":
        if num_classes == 100:
            return get_resnet50_cifar100()
        else:
            from dacp.models.resnet import ResNet50ForCIFAR

            return ResNet50ForCIFAR(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_data_loaders(dataset_name: str, batch_size: int, data_dir: str, num_workers: int):
    """
    获取数据加载器。

    参数:
        dataset_name: 数据集名称 ('cifar10', 'cifar100')
        batch_size: 批次大小
        data_dir: 数据目录
        num_workers: 工作进程数

    返回:
        (train_loader, test_loader) 元组
    """
    if dataset_name == "cifar10":
        return get_cifar10_loaders(
            batch_size=batch_size,
            data_dir=data_dir,
            num_workers=num_workers,
        )
    elif dataset_name == "cifar100":
        return get_cifar100_loaders(
            batch_size=batch_size,
            data_dir=data_dir,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Train CV models")

    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Model architecture",
    )

    # 数据集参数
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset name",
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")

    # 学习率调度
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "step", "none"],
        help="Learning rate schedule",
    )
    parser.add_argument(
        "--lr_step_size", type=int, default=50, help="Step size for step LR schedule"
    )
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Gamma for step LR schedule")

    # 检查点参数
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory"
    )
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # 早停参数
    parser.add_argument(
        "--early_stopping", type=int, default=None, help="Early stopping patience (None to disable)"
    )

    # 其他参数
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use"
    )
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps"
    )

    args = parser.parse_args()

    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 确定类别数
    num_classes = 10 if args.dataset == "cifar10" else 100

    # 创建模型
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_classes)

    # 加载数据
    print(f"Loading dataset: {args.dataset}")
    train_loader, val_loader = get_data_loaders(
        args.dataset,
        args.batch_size,
        args.data_dir,
        args.num_workers,
    )

    # 创建优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # 创建学习率调度器
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    elif args.lr_schedule == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
        )

    # 创建损失函数
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
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print("=" * 60)

    history = trainer.train(
        epochs=args.epochs,
        save_every=args.save_every,
        verbose=True,
        start_epoch=start_epoch,
    )

    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
