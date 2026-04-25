"""
ViT-L/32 在 ImageNet 上微调的脚本。

运行命令:
    python experiments/scripts/finetune/finetune_vit.py \
        --total_steps 1000 --batch_size 32 \
        --save_every 200 --device cuda \
        --output_dir checkpoints/vit_l32_imagenet \
        --data_dir ./data

    # 使用 ViT-B/16
    python experiments/scripts/finetune/finetune_vit.py \
        --model vit-b-16 --total_steps 1000 --batch_size 64 \
        --output_dir checkpoints/vit_b16_imagenet --device cuda

特点:
- 支持 ViT-L/32 和 ViT-B/16
- 在 ImageNet-1K 上微调
- 基于步数训练，每 N 步保存检查点（含优化器状态）
- 支持混合精度训练 (AMP)
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import torch
import torch.nn as nn
from pathlib import Path
import sys
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from transformers import ViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_imagenet_loaders(batch_size, num_workers, data_dir):
    """获取 ImageNet-1K 训练和验证 DataLoader。"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_size = int(os.environ.get('IMAGE_SIZE', 384))
    resize_size = int(image_size * 256 / 224)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(data_dir, 'imagenet', 'train')
    val_dir = os.path.join(data_dir, 'imagenet', 'val')

    if os.path.isdir(train_dir):
        train_ds = datasets.ImageFolder(train_dir, train_transform)
        val_ds = datasets.ImageFolder(val_dir, val_transform)
    else:
        from datasets import load_dataset as hf_load

        raw = hf_load('imagenet-1k')

        class HFImageNet(torch.utils.data.Dataset):
            def __init__(self, hf_split, transform):
                self.data = hf_split
                self.transform = transform
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                item = self.data[idx]
                img = item['image'].convert('RGB')
                return self.transform(img), item['label']

        train_ds = HFImageNet(raw['train'], train_transform)
        val_ds = HFImageNet(raw['validation'], val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def train_step(model, optimizer, scaler, images, labels, device, use_amp=True):
    """训练一步，返回 loss 值。"""
    model.train()
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    if use_amp:
        with torch.cuda.amp.autocast():
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, val_loader, device):
    """在验证集上评估 top-1 准确率。"""
    model.eval()
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(pixel_values=images)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


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


MODEL_HF_MAP = {
    'vit-l-32': 'google/vit-large-patch32-384',
    'vit-b-16': 'google/vit-base-patch16-224',
}


def main():
    parser = argparse.ArgumentParser(description='ViT ImageNet 微调脚本')
    parser.add_argument('--model', type=str, default='vit-l-32',
                        choices=list(MODEL_HF_MAP.keys()),
                        help='ViT 模型变体 (默认: vit-l-32)')
    parser.add_argument('--total_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/vit_l32_imagenet')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=True)

    args = parser.parse_args()

    hf_name = MODEL_HF_MAP[args.model]

    print("=" * 60)
    print(f"{args.model.upper()} ImageNet 微调")
    print("=" * 60)
    print(f"HuggingFace 模型: {hf_name}")
    print(f"总步数: {args.total_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"混合精度: {args.use_amp}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    # 加载模型
    print(f"\n加载 {args.model} 预训练模型...")
    model = ViTForImageClassification.from_pretrained(hf_name)
    model = model.to(args.device)
    print(f"模型已加载到 {args.device}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # 加载数据
    print("\n加载 ImageNet-1K 数据集...")
    train_loader, val_loader = get_imagenet_loaders(
        args.batch_size, args.num_workers, args.data_dir)
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")

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
            model, optimizer, scaler, images, labels,
            args.device, args.use_amp)
        running_loss += loss

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            print(f"\nStep {step}/{args.total_steps}: Loss={avg_loss:.4f}")
            running_loss = 0.0

        if step % args.save_every == 0:
            print(f"\n保存检查点 (step {step})...")
            save_checkpoint(model, optimizer, step, loss, args.output_dir)

    # 保存最终检查点
    print("\n保存最终检查点...")
    save_checkpoint(model, optimizer, args.total_steps, loss, args.output_dir)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"检查点已保存到: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
