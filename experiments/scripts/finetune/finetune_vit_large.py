"""
ViT-L/32 在 ImageNet-1K 上微调的脚本。

运行命令:
    python experiments/scripts/finetune/finetune_vit_large.py \
        --total_steps 1000 --batch_size 16 --save_every 200 \
        --device cuda --output_dir checkpoints/vit_large_imagenet

特点:
- 使用 google/vit-large-patch32-384 预训练模型
- 在 ImageNet-1K 上微调（1000类分类）
- Warmup + cosine decay LR schedule
- 支持混合精度训练 (AMP)
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import math
from pathlib import Path
import sys
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from transformers import ViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

LOCAL_MODEL_PATH = "/lihongliang/fangzl/ckpt-compress/data/models/models--google--vit-large-patch32-384/snapshots/a2b30ad36d02e99f045cd2ecfc71e0ae16991efa"
IMAGENET_PATH = "/lihongliang/bobzhou/dataset/imagenet"


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Warmup + cosine decay 学习率调度器。"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def prepare_imagenet_loaders(image_size, batch_size, num_workers=4):
    """准备 ImageNet-1K 的 train/val DataLoader。"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.143)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dir = os.path.join(IMAGENET_PATH, 'train')
    val_dir = os.path.join(IMAGENET_PATH, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def train_step(model, optimizer, scaler, images, labels, device, use_amp=True):
    """训练一步，返回 loss 值。"""
    model.train()
    criterion = nn.CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    if use_amp:
        with torch.cuda.amp.autocast():
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_acc(model, val_loader, device, max_batches=50, use_amp=True):
    """在验证集上计算 Top-1 Accuracy。"""
    model.eval()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(val_loader):
        if i >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=images)
        else:
            outputs = model(pixel_values=images)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc


def save_checkpoint(model, optimizer, step, loss, output_dir):
    """保存检查点（含优化器状态）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f'checkpoint_step_{step}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss,
    }, checkpoint_path)
    print(f"  检查点已保存: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='ViT-L/32 ImageNet 微调脚本')
    parser.add_argument('--total_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/vit_large_imagenet')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    print("=" * 60)
    print("ViT-L/32 ImageNet 微调")
    print("=" * 60)
    print(f"总步数: {args.total_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"图片大小: {args.image_size}")
    print(f"学习率: {args.lr}")
    print(f"混合精度: {args.use_amp}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    # 加载模型
    print("\n加载 ViT-L/32 预训练模型...")
    model = ViTForImageClassification.from_pretrained(
        LOCAL_MODEL_PATH, num_labels=1000, ignore_mismatched_sizes=True)
    model = model.to(args.device)
    print(f"模型已加载到 {args.device}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 创建优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = max(1, args.total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.total_steps)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    print(f"LR schedule: warmup {warmup_steps} steps + cosine decay")

    # 加载数据
    print(f"\n加载 ImageNet-1K 数据集 ({IMAGENET_PATH})...")
    train_loader, val_loader = prepare_imagenet_loaders(
        args.image_size, args.batch_size, args.num_workers)
    print(f"训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")

    # 初始 eval
    init_acc = evaluate_acc(model, val_loader, args.device, max_batches=50, use_amp=args.use_amp)
    print(f"\n初始 Val Acc: {init_acc:.4f} ({init_acc*100:.2f}%)")

    # 训练循环
    print("\n开始训练...")
    print("=" * 60)

    data_iter = iter(train_loader)
    running_loss = 0.0
    log_interval = 10

    for step in tqdm(range(1, args.total_steps + 1), desc="Training"):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        loss = train_step(
            model, optimizer, scaler, images, labels, args.device, args.use_amp)
        scheduler.step()
        running_loss += loss

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            lr = scheduler.get_last_lr()[0]
            print(f"\nStep {step}/{args.total_steps}: "
                  f"Loss={avg_loss:.4f}, LR={lr:.2e}")
            running_loss = 0.0

        if step % args.save_every == 0:
            val_acc = evaluate_acc(model, val_loader, args.device,
                                   max_batches=50, use_amp=args.use_amp)
            print(f"  Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"  保存检查点 (step {step})...")
            save_checkpoint(model, optimizer, step, loss, args.output_dir)

    # 保存最终检查点
    print("\n保存最终检查点...")
    save_checkpoint(model, optimizer, args.total_steps, loss, args.output_dir)

    # 最终评估
    final_acc = evaluate_acc(model, val_loader, args.device, max_batches=100, use_amp=args.use_amp)
    print(f"\n最终 Val Acc: {final_acc:.4f} ({final_acc*100:.2f}%)")

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"检查点已保存到: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
