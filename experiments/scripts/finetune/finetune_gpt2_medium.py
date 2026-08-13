"""
GPT-2 Medium 微调脚本。

在 WikiText-103 上微调 GPT-2 Medium，生成带优化器状态的检查点。

nohup python3 experiments/scripts/finetune/finetune_gpt2_medium.py \
         --total_steps 1000 --batch_size 4 --seq_length 512 \
         --save_every 200 --device cuda \
         --output_dir checkpoints/gpt2_medium_wikitext103 \
         > finetune_gpt2m.log 2>&1 &

运行命令:
    python experiments/scripts/finetune/finetune_gpt2_medium.py \
        --total_steps 1000 \
        --batch_size 4 \
        --seq_length 512 \
        --lr 2e-5 \
        --weight_decay 0.01 \
        --save_every 200 \
        --device cuda \
        --output_dir checkpoints/gpt2_medium_wikitext103

特点:
    - 使用混合精度训练加速
    - 每 N 步保存检查点（包含优化器状态）
    - 打印训练进度和 loss
    - 自动处理磁盘空间检查问题
"""

import os

os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from transformers import GPT2LMHeadModel

from dacp.utils.data_loader import get_wikitext103_dataloader
from dacp.utils.paths import resolve_data_file, resolve_model_source
from experiments.lib.losses import causal_lm_loss, move_batch_to_device, perplexity_from_loss

LOCAL_MODEL_PATH = os.environ.get(
    "GPT2M_MODEL_PATH",
    resolve_model_source("gpt2-medium", "gpt2-medium"),
)
LOCAL_WIKITEXT_PATH = str(resolve_data_file("wikitext103"))


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Warmup + cosine decay 学习率调度器。"""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_ppl(model, eval_loader, device, max_batches=50):
    """在 eval 数据上计算 PPL。"""
    if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1:
        raise ValueError(f"max_batches must be a positive integer, got {max_batches}")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break
            batch_on_device = move_batch_to_device(batch, device)
            loss, supervised_tokens = causal_lm_loss(
                model,
                batch_on_device,
                reduction="sum",
            )
            total_tokens += supervised_tokens
            total_loss += float(loss.item())
        if total_tokens == 0:
            raise ValueError("evaluation requires at least one supervised causal-LM token")
        return perplexity_from_loss(total_loss / total_tokens)
    finally:
        model.train(was_training)


def train_step(model, optimizer, scaler, batch, device, use_amp=True):
    """
    训练一步。

    参数:
        model: GPT-2 Medium 模型
        optimizer: 优化器
        scaler: GradScaler（用于混合精度训练）
        batch: 数据批次
        device: 设备
        use_amp: 是否使用混合精度训练

    返回:
        loss: 损失值
    """
    model.train()
    batch_on_device = move_batch_to_device(batch, device)

    optimizer.zero_grad()

    if use_amp:
        # 混合精度训练
        with torch.cuda.amp.autocast():
            loss, _ = causal_lm_loss(model, batch_on_device)

        # Backward with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        # 标准训练
        loss, _ = causal_lm_loss(model, batch_on_device)

        loss.backward()
        optimizer.step()

    return loss.item()


def save_checkpoint(model, optimizer, step, loss, output_dir):
    """
    保存检查点。

    参数:
        model: 模型
        optimizer: 优化器
        step: 当前步数
        loss: 当前损失
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": loss,
        },
        checkpoint_path,
    )

    print(f"  检查点已保存: {checkpoint_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GPT-2 Medium 微调脚本")
    parser.add_argument("--total_steps", type=int, default=1000, help="总训练步数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--seq_length", type=int, default=512, help="序列长度")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--save_every", type=int, default=200, help="每 N 步保存检查点")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints/gpt2_medium_wikitext103", help="输出目录"
    )
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用混合精度训练（可用 --no-use_amp 关闭）",
    )

    args = parser.parse_args(argv)
    args.use_amp = bool(args.use_amp and str(args.device).startswith("cuda"))
    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("GPT-2 Medium 微调")
    print("=" * 60)
    print(f"总步数: {args.total_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"序列长度: {args.seq_length}")
    print(f"学习率: {args.lr}")
    print(f"权重衰减: {args.weight_decay}")
    print(f"保存间隔: 每 {args.save_every} 步")
    print(f"混合精度: {args.use_amp}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    # 加载模型
    print("\n加载 GPT-2 Medium 预训练模型...")
    model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    model = model.to(args.device)
    print(f"模型已加载到 {args.device}")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 创建学习率调度器（warmup + cosine decay）
    warmup_steps = args.total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.total_steps)
    print(f"LR Schedule: warmup {warmup_steps} steps + cosine decay")

    # 创建 GradScaler（用于混合精度训练）
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # 加载数据
    print("\n加载 WikiText-103 训练数据...")
    train_loader = get_wikitext103_dataloader(
        split="train",
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0,
        shuffle=True,
        local_path=os.environ.get("WIKITEXT103_PATH", LOCAL_WIKITEXT_PATH),
    )
    print(f"训练数据加载器已创建")

    # 加载 validation 数据
    print("加载 WikiText-103 validation 数据...")
    eval_loader = get_wikitext103_dataloader(
        split="validation",
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0,
        shuffle=False,
        local_path=os.environ.get("WIKITEXT103_PATH", LOCAL_WIKITEXT_PATH),
    )
    print(f"Validation 数据加载器已创建")

    # 评估初始 PPL
    init_ppl = evaluate_ppl(model, eval_loader, args.device)
    print(f"\n初始 Validation PPL: {init_ppl:.4f}")

    # 训练循环
    print("\n开始训练...")
    print("=" * 60)

    data_iter = iter(train_loader)
    running_loss = 0.0
    log_interval = 10

    for step in tqdm(range(1, args.total_steps + 1), desc="Training"):
        # 获取批次
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # 训练一步
        loss = train_step(model, optimizer, scaler, batch, args.device, args.use_amp)
        scheduler.step()
        running_loss += loss

        # 打印进度
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"\nStep {step}/{args.total_steps}: Loss={avg_loss:.4f}, TrainPPL={perplexity:.2f}, LR={current_lr:.2e}"
            )
            running_loss = 0.0

        # 保存检查点 + 评估
        if step % args.save_every == 0:
            val_ppl = evaluate_ppl(model, eval_loader, args.device)
            print(f"\n[Step {step}] Validation PPL: {val_ppl:.4f}")
            print(f"保存检查点 (step {step})...")
            save_checkpoint(model, optimizer, step, loss, args.output_dir)

    # 保存最终检查点
    print("\n保存最终检查点...")
    save_checkpoint(model, optimizer, args.total_steps, loss, args.output_dir)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"检查点已保存到: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
