"""
Pythia-410M 在 Alpaca 52K 上微调的脚本。

运行命令:
    python experiments/scripts/finetune/finetune_pythia_410m.py \
        --total_steps 1000 --batch_size 4 --seq_length 512 \
        --save_every 200 --device cuda \
        --output_dir checkpoints/pythia_410m_alpaca

特点:
- 使用 EleutherAI/pythia-410m 预训练模型
- 在 Alpaca 52K 指令微调数据集上训练
- 基于步数训练，每 N 步保存检查点（含优化器状态）
- 支持混合精度训练 (AMP)
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

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from dacp.utils.paths import resolve_data_file, resolve_model_source
from experiments.lib.data import _alpaca_split_sizes, tokenize_alpaca_batch
from experiments.lib.losses import causal_lm_loss, move_batch_to_device, perplexity_from_loss

LOCAL_MODEL_PATH = os.environ.get(
    "PYTHIA_MODEL_PATH",
    resolve_model_source("pythia-410m", "EleutherAI/pythia-410m"),
)
LOCAL_ALPACA_PATH = str(resolve_data_file("alpaca/alpaca_data.json"))


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Warmup + cosine decay 学习率调度器。"""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_ppl(model, val_loader, device, max_batches=50):
    """在验证集上计算 PPL。"""
    if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1:
        raise ValueError(f"max_batches must be a positive integer, got {max_batches}")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            batch_on_device = move_batch_to_device(batch, device)
            loss, supervised_tokens = causal_lm_loss(
                model,
                batch_on_device,
                reduction="sum",
            )
            total_tokens += supervised_tokens
            total_loss += loss.item()
        if total_tokens == 0:
            raise ValueError("validation batches contain no supervised causal-LM tokens")
        avg_loss = total_loss / total_tokens
        return perplexity_from_loss(avg_loss), avg_loss
    finally:
        model.train(was_training)


def prepare_alpaca_loaders(tokenizer, batch_size, seq_length, num_workers=0):
    """准备 Alpaca 52K 数据集的 DataLoader。"""
    import json

    local_path = os.environ.get("ALPACA_LOCAL_PATH", LOCAL_ALPACA_PATH)
    if local_path and os.path.isfile(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
        print(f"  Alpaca: loaded {len(dataset)} samples from local JSON")
    else:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")

    def format_and_tokenize(examples):
        return tokenize_alpaca_batch(examples, tokenizer, seq_length)

    dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")

    n_train, n_val = _alpaca_split_sizes(len(dataset))
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def train_step(model, optimizer, scaler, batch, device, use_amp=True):
    """训练一步，返回 loss 值。"""
    model.train()
    batch_on_device = move_batch_to_device(batch, device)

    optimizer.zero_grad()

    if use_amp:
        with torch.cuda.amp.autocast():
            loss, _ = causal_lm_loss(model, batch_on_device)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss, _ = causal_lm_loss(model, batch_on_device)
        loss.backward()
        optimizer.step()

    return loss.item()


def save_checkpoint(model, optimizer, step, loss, output_dir):
    """保存检查点（含优化器状态）。"""
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
    parser = argparse.ArgumentParser(description="Pythia-410M Alpaca 微调脚本")
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="checkpoints/pythia_410m_alpaca")
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args(argv)
    args.use_amp = bool(args.use_amp and str(args.device).startswith("cuda"))
    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("Pythia-410M Alpaca 微调")
    print("=" * 60)
    print(f"总步数: {args.total_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"序列长度: {args.seq_length}")
    print(f"学习率: {args.lr}")
    print(f"混合精度: {args.use_amp}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    # 加载模型
    print("\n加载 Pythia-410M 预训练模型...")
    model = GPTNeoXForCausalLM.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.float32)
    model = model.to(args.device)
    print(f"模型已加载到 {args.device}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = max(1, args.total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.total_steps)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    print(f"LR schedule: warmup {warmup_steps} steps + cosine decay")

    # 加载数据
    print("\n加载 Alpaca 52K 数据集...")
    train_loader, val_loader = prepare_alpaca_loaders(tokenizer, args.batch_size, args.seq_length)
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")

    # 初始 eval
    init_ppl, init_loss = evaluate_ppl(model, val_loader, args.device)
    print(f"\n初始 Val PPL: {init_ppl:.2f}, Loss: {init_loss:.4f}")

    # 训练循环
    print("\n开始训练...")
    print("=" * 60)

    data_iter = iter(train_loader)
    running_loss = 0.0
    log_interval = 10

    for step in tqdm(range(1, args.total_steps + 1), desc="Training"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        loss = train_step(model, optimizer, scaler, batch, args.device, args.use_amp)
        scheduler.step()
        running_loss += loss

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            ppl = math.exp(avg_loss)
            lr = scheduler.get_last_lr()[0]
            print(
                f"\nStep {step}/{args.total_steps}: "
                f"Loss={avg_loss:.4f}, PPL={ppl:.2f}, LR={lr:.2e}"
            )
            running_loss = 0.0

        if step % args.save_every == 0:
            val_ppl, val_loss = evaluate_ppl(model, val_loader, args.device)
            print(f"  Val PPL: {val_ppl:.2f}, Val Loss: {val_loss:.4f}")
            print(f"  保存检查点 (step {step})...")
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
