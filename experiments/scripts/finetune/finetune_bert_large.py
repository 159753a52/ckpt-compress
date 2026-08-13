"""
BERT-Large 在 GLUE 任务（SST-2 / MNLI / STS-B）上微调的统一脚本。

运行命令:
    # SST-2（情感分类）
    python experiments/scripts/finetune/finetune_bert_large.py \
        --dataset sst2 --num_steps 1000 --batch_size 16 \
        --checkpoint_dir checkpoints/bert_large_sst2_1000steps --device cuda

    # MNLI（自然语言推理）
    python experiments/scripts/finetune/finetune_bert_large.py \
        --dataset mnli --num_steps 1000 --batch_size 16 \
        --checkpoint_dir checkpoints/bert_large_mnli_1000steps --device cuda

    # STS-B（语义相似度回归）
    python experiments/scripts/finetune/finetune_bert_large.py \
        --dataset stsb --num_steps 1000 --batch_size 16 \
        --checkpoint_dir checkpoints/bert_large_stsb_1000steps --device cuda

特点:
- 基于步数训练（不是 epoch）
- 支持混合精度训练 (AMP)
- 支持梯度累积
- 每 200 步保存一次检查点
- 自动选择评估指标（SST-2/MNLI: 准确率，STS-B: 皮尔逊相关系数）

替代原先的 finetune_bert_large_sst2.py / finetune_bert_large_mnli.py / finetune_bert_large_stsb.py。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from transformers import BertForSequenceClassification

from dacp.utils.data_loader import get_mnli_loaders, get_sst2_loaders, get_stsb_loaders
from dacp.utils.paths import resolve_model_source
from experiments.lib.args import nonnegative_int, positive_int

# 数据集配置
DATASET_CONFIG = {
    "sst2": {"num_labels": 2, "task": "classification", "metric_name": "准确率"},
    "mnli": {"num_labels": 3, "task": "classification", "metric_name": "准确率"},
    "qqp": {"num_labels": 2, "task": "classification", "metric_name": "准确率"},
    "qnli": {"num_labels": 2, "task": "classification", "metric_name": "准确率"},
    "mrpc": {"num_labels": 2, "task": "classification", "metric_name": "准确率"},
    "cola": {"num_labels": 2, "task": "classification", "metric_name": "准确率"},
    "rte": {"num_labels": 2, "task": "classification", "metric_name": "准确率"},
    "stsb": {"num_labels": 1, "task": "regression", "metric_name": "相关系数"},
}


def evaluate_classification(model, eval_loader, device):
    """在验证集上评估准确率（SST-2 / MNLI）。"""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += int((predictions == labels).sum().item())
                total += labels.size(0)
        if total == 0:
            raise ValueError("classification evaluation requires at least one example")
        return correct / total
    finally:
        model.train(was_training)


def evaluate_regression(model, eval_loader, device):
    """在验证集上评估皮尔逊相关系数（STS-B）。"""
    was_training = model.training
    model.eval()
    all_preds = []
    all_labels = []
    try:
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                preds = outputs.logits.squeeze(-1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        if not all_preds:
            raise ValueError("regression evaluation requires at least one batch")
        stacked_preds = torch.cat(all_preds).double()
        stacked_labels = torch.cat(all_labels).double()
        if stacked_preds.numel() < 2:
            raise ValueError("Pearson evaluation requires at least two examples")
        centered_preds = stacked_preds - stacked_preds.mean()
        centered_labels = stacked_labels - stacked_labels.mean()
        denominator = centered_preds.norm() * centered_labels.norm()
        if denominator <= 0:
            raise ValueError("Pearson evaluation requires non-constant predictions and labels")
        correlation = float((centered_preds * centered_labels).sum().item() / denominator.item())
        if not torch.isfinite(torch.tensor(correlation)):
            raise ValueError("Pearson evaluation produced a non-finite correlation")
        if correlation < -1.0 - 1e-12 or correlation > 1.0 + 1e-12:
            raise ValueError("Pearson evaluation produced an out-of-range correlation")
        return max(-1.0, min(1.0, correlation))
    finally:
        model.train(was_training)


def train_n_steps(
    model,
    train_loader,
    eval_loader,
    optimizer,
    device,
    task_type,
    metric_name,
    num_steps=1000,
    gradient_accumulation_steps=1,
    use_amp=False,
    log_interval=50,
    save_interval=200,
    checkpoint_dir="./checkpoints/bert_large",
):
    """训练指定步数。"""
    evaluate_fn = evaluate_classification if task_type == "classification" else evaluate_regression
    model.train()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    total_loss = 0.0
    step = 0
    best_metric = 0.0
    eval_metric = 0.0
    optimizer.zero_grad()

    history: Dict[str, List[float | int]] = {
        "steps": [],
        "train_loss": [],
        f"eval_{metric_name}": [],
    }

    data_iter = iter(train_loader)

    print(f"开始训练 {num_steps} 个 optimizer steps...")
    print(f"设备: {device}")
    print(f"梯度累积步数: {gradient_accumulation_steps}")
    print(f"混合精度: {use_amp}")
    print(f"评估指标: {metric_name}")
    print(f"检查点目录: {checkpoint_dir}")
    print("=" * 80)

    start_time = time.time()

    micro_steps = 0
    with tqdm(total=num_steps, desc="训练进度") as pbar:
        while step < num_steps:
            accumulated_loss = 0.0
            for _ in range(gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                    loss.backward()
                accumulated_loss += loss.item()
                micro_steps += 1

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            step += 1
            total_loss += accumulated_loss
            pbar.update(1)

            if step % log_interval == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                eta = (num_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "step/s": f"{steps_per_sec:.2f}",
                        "ETA": f"{eta/60:.1f}min",
                    }
                )
                total_loss = 0.0

            if step % save_interval == 0 or step == num_steps:
                if eval_loader is not None:
                    eval_metric = evaluate_fn(model, eval_loader, device)
                    history["steps"].append(step)
                    history["train_loss"].append(avg_loss if step % log_interval == 0 else 0)
                    history[f"eval_{metric_name}"].append(eval_metric)

                    print(f"\nStep {step}: 验证{metric_name} = {eval_metric:.4f}")

                    if eval_metric > best_metric:
                        best_metric = eval_metric
                        best_path = checkpoint_dir / "checkpoint_best.pt"
                        _save_checkpoint(
                            model,
                            optimizer,
                            step,
                            micro_steps,
                            gradient_accumulation_steps,
                            eval_metric,
                            metric_name,
                            best_path,
                        )
                        print(f"最佳模型已保存: {best_path} " f"({metric_name}: {eval_metric:.4f})")

                ckpt_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
                _save_checkpoint(
                    model,
                    optimizer,
                    step,
                    micro_steps,
                    gradient_accumulation_steps,
                    eval_metric if eval_loader is not None else None,
                    metric_name,
                    ckpt_path,
                )
                print(f"检查点已保存: {ckpt_path}")

    final_path = checkpoint_dir / f"checkpoint_step_{num_steps}_final.pt"
    _save_checkpoint(
        model,
        optimizer,
        num_steps,
        micro_steps,
        gradient_accumulation_steps,
        eval_metric if eval_loader is not None else None,
        metric_name,
        final_path,
    )

    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"训练完成！")
    print(f"总步数: {num_steps}")
    print(f"总时间: {elapsed_time/60:.2f} 分钟")
    print(f"平均速度: {num_steps/elapsed_time:.2f} 步/秒")
    print(f"最佳验证{metric_name}: {best_metric:.4f}")
    print(f"最终检查点: {final_path}")
    print(f"训练历史: {history_path}")


def _save_checkpoint(
    model,
    optimizer,
    step,
    micro_steps,
    gradient_accumulation_steps,
    eval_metric,
    metric_name,
    path,
):
    """保存检查点。"""
    checkpoint = {
        "step": step,
        "optimizer_steps": step,
        "micro_steps": micro_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if eval_metric is not None:
        checkpoint[f"eval_{metric_name}"] = eval_metric
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="BERT-Large 在 GLUE 任务上微调")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIG.keys()),
        help="GLUE 数据集名称",
    )

    # 训练参数
    parser.add_argument("--num_steps", type=positive_int, default=1000)
    parser.add_argument("--batch_size", type=positive_int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=positive_int, default=2)
    parser.add_argument("--max_length", type=positive_int, default=128)

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--train_subset", type=positive_int, default=None)
    parser.add_argument("--val_subset", type=positive_int, default=None)
    parser.add_argument("--no_eval", action="store_true")
    # MNLI 专属
    parser.add_argument(
        "--matched", action="store_true", help="[MNLI] 使用 matched 验证集（默认 mismatched）"
    )

    # 检查点参数
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None, help="检查点保存目录（默认自动生成）"
    )
    parser.add_argument("--save_interval", type=positive_int, default=200)
    parser.add_argument("--log_interval", type=positive_int, default=50)

    # 其他参数
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--num_workers", type=nonnegative_int, default=4)

    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"./checkpoints/bert_large_{args.dataset}_1000steps"

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")

    print("=" * 80)
    print(f"BERT-Large 在 {args.dataset.upper()} 上微调")
    print("=" * 80)
    print(f"训练步数: {args.num_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大序列长度: {args.max_length}")
    print(f"学习率: {args.lr}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"设备: {device}")
    print(f"混合精度: {args.use_amp}")
    print(f"任务类型: {cfg['task']} ({cfg['metric_name']})")
    print("=" * 80)

    # 创建模型
    print("\n加载 BERT-Large 模型...")
    model = BertForSequenceClassification.from_pretrained(
        os.environ.get(
            "BERT_MODEL_PATH",
            resolve_model_source("bert-large-uncased", "bert-large-uncased"),
        ),
        num_labels=cfg["num_labels"],
        local_files_only=False,
    )
    if cfg["task"] == "regression":
        model.config.problem_type = "regression"
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")
    print(f"可训练参数: {num_trainable:,}")

    # 加载数据
    print(f"\n加载 {args.dataset.upper()} 数据集...")
    loader_kwargs = dict(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    if args.dataset == "sst2":
        train_loader, val_loader = get_sst2_loaders(
            **loader_kwargs,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
        )
    elif args.dataset == "mnli":
        train_loader, val_matched, val_mismatched = get_mnli_loaders(
            **loader_kwargs,
            train_subset=args.train_subset,
            val_matched_subset=args.val_subset,
            val_mismatched_subset=args.val_subset,
        )
        val_loader = val_matched if args.matched else val_mismatched
        print(f"验证集: {'matched' if args.matched else 'mismatched'}")
    elif args.dataset == "stsb":
        train_loader, val_loader = get_stsb_loaders(
            **loader_kwargs,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
        )
    else:
        # 通用 GLUE 任务（QQP, QNLI, MRPC, CoLA, RTE）
        from experiments.lib.data import get_data_loaders

        train_loader, val_loader, task_type = get_data_loaders(
            "bert-large",
            args.dataset,
            batch_size=args.batch_size,
            seq_length=args.max_length,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
        )
        if task_type != "cls":
            raise RuntimeError(f"Unexpected task type for {args.dataset}: {task_type}")

    if args.no_eval:
        val_loader = None

    print(f"训练数据批次数: {len(train_loader)}")
    if val_loader is not None:
        print(f"验证数据批次数: {len(val_loader)}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("\n" + "=" * 80)
    train_n_steps(
        model=model,
        train_loader=train_loader,
        eval_loader=val_loader,
        optimizer=optimizer,
        device=device,
        task_type=cfg["task"],
        metric_name=cfg["metric_name"],
        num_steps=args.num_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
