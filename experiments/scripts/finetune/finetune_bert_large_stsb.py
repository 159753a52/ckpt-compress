"""
BERT-Large 在 STS-B 上微调的脚本。

运行命令:
    python experiments/scripts/finetune_bert_large_stsb.py \
        --num_steps 1000 \
        --batch_size 16 \
        --gradient_accumulation_steps 2 \
        --learning_rate 2e-5 \
        --save_every 200 \
        --output_dir checkpoints/bert_large_stsb_1000steps \
        --device cuda

特点:
- 基于步数训练（不是 epoch）
- 支持混合精度训练 (AMP)
- 支持梯度累积
- 每 200 步保存一次检查点
- 支持在验证集上评估皮尔逊相关系数
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import sys
import time
from tqdm import tqdm
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.utils.data_loader import get_stsb_loaders
from transformers import BertForSequenceClassification


def evaluate(model, eval_loader, device):
    """在验证集上评估皮尔逊相关系数"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # STS-B 是回归任务，logits 的形状是 [batch_size, 1]
            preds = logits.squeeze(-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # 计算皮尔逊相关系数
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # 使用 torch.corrcoef 计算相关系数
    stacked = torch.stack([all_preds, all_labels])
    correlation_matrix = torch.corrcoef(stacked)
    correlation = correlation_matrix[0, 1].item()

    model.train()
    return correlation


def train_n_steps(
    model,
    train_loader,
    eval_loader,
    optimizer,
    device,
    num_steps=1000,
    gradient_accumulation_steps=1,
    use_amp=False,
    log_interval=50,
    save_interval=200,
    checkpoint_dir='./checkpoints/bert_large_stsb_1000steps',
):
    """
    训练指定步数。

    参数:
        model: 模型
        train_loader: 训练数据加载器
        eval_loader: 验证数据加载器
        optimizer: 优化器
        device: 设备
        num_steps: 训练步数
        gradient_accumulation_steps: 梯度累积步数
        use_amp: 是否使用混合精度
        log_interval: 日志打印间隔
        save_interval: 检查点保存间隔
        checkpoint_dir: 检查点保存目录
    """
    model.train()

    # 创建检查点目录
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 训练统计
    total_loss = 0.0
    step = 0
    best_corr = 0.0
    optimizer.zero_grad()

    # 训练历史
    history = {
        'steps': [],
        'train_loss': [],
        'eval_corr': [],
    }

    # 创建数据迭代器
    data_iter = iter(train_loader)

    print(f"开始训练 {num_steps} 步...")
    print(f"设备: {device}")
    print(f"梯度累积步数: {gradient_accumulation_steps}")
    print(f"混合精度: {use_amp}")
    print(f"检查点目录: {checkpoint_dir}")
    print("=" * 80)

    start_time = time.time()

    with tqdm(total=num_steps, desc="训练进度") as pbar:
        while step < num_steps:
            try:
                # 获取下一个批次
                batch = next(data_iter)
            except StopIteration:
                # 数据集遍历完毕，重新开始
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # 将数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / gradient_accumulation_steps

                # 反向传播
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps

                # 反向传播
                loss.backward()

            total_loss += loss.item() * gradient_accumulation_steps

            # 梯度累积
            if (step + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            step += 1
            pbar.update(1)

            # 日志打印
            if step % log_interval == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                eta = (num_steps - step) / steps_per_sec if steps_per_sec > 0 else 0

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'step/s': f'{steps_per_sec:.2f}',
                    'ETA': f'{eta/60:.1f}min'
                })

                total_loss = 0.0

            # 保存检查点和评估
            if step % save_interval == 0 or step == num_steps:
                # 在验证集上评估
                eval_corr = 0.0
                if eval_loader is not None:
                    eval_corr = evaluate(model, eval_loader, device)
                    history['steps'].append(step)
                    history['train_loss'].append(avg_loss if step % log_interval == 0 else 0)
                    history['eval_corr'].append(eval_corr)

                    print(f"\nStep {step}: 验证相关系数 = {eval_corr:.4f}")

                    # 保存最佳模型
                    if eval_corr > best_corr:
                        best_corr = eval_corr
                        best_checkpoint_path = checkpoint_dir / 'checkpoint_best.pt'
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            eval_corr=eval_corr,
                            checkpoint_path=best_checkpoint_path,
                        )
                        print(f"最佳模型已保存: {best_checkpoint_path} (相关系数: {eval_corr:.4f})")

                # 保存定期检查点
                checkpoint_path = checkpoint_dir / f'checkpoint_step_{step}.pt'
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    eval_corr=eval_corr if eval_loader is not None else None,
                    checkpoint_path=checkpoint_path,
                )
                print(f"检查点已保存: {checkpoint_path}")

    # 保存最终检查点
    final_checkpoint_path = checkpoint_dir / f'checkpoint_step_{num_steps}_final.pt'
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=num_steps,
        eval_corr=eval_corr if eval_loader is not None else None,
        checkpoint_path=final_checkpoint_path,
    )

    # 保存训练历史
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"训练完成！")
    print(f"总步数: {num_steps}")
    print(f"总时间: {elapsed_time/60:.2f} 分钟")
    print(f"平均速度: {num_steps/elapsed_time:.2f} 步/秒")
    print(f"最佳验证相关系数: {best_corr:.4f}")
    print(f"最终检查点: {final_checkpoint_path}")
    print(f"训练历史: {history_path}")


def save_checkpoint(model, optimizer, step, eval_corr, checkpoint_path):
    """保存检查点。"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if eval_corr is not None:
        checkpoint['eval_corr'] = eval_corr
    torch.save(checkpoint, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(
        description='BERT-Large 在 STS-B 上微调'
    )

    # 训练参数
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='训练步数 (默认: 1000)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小 (默认: 16)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='学习率 (默认: 2e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减 (默认: 0.01)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='梯度累积步数 (默认: 2)')
    parser.add_argument('--max_length', type=int, default=128,
                        help='最大序列长度 (默认: 128)')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='本地数据目录 (默认: ./data)')
    parser.add_argument('--train_subset', type=int, default=None,
                        help='训练集子集大小 (可选，用于快速测试)')
    parser.add_argument('--val_subset', type=int, default=None,
                        help='验证集子集大小 (可选，用于快速测试)')

    # 检查点参数
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./checkpoints/bert_large_stsb_1000steps',
                        help='检查点保存目录')
    parser.add_argument('--save_interval', type=int, default=200,
                        help='检查点保存间隔 (默认: 200)')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='日志打印间隔 (默认: 50)')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备 (默认: cuda)')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数 (默认: 4)')

    args = parser.parse_args()

    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")

    print("=" * 80)
    print("BERT-Large 在 STS-B 上微调")
    print("=" * 80)
    print(f"训练步数: {args.num_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大序列长度: {args.max_length}")
    print(f"学习率: {args.lr}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"设备: {device}")
    print(f"混合精度: {args.use_amp}")
    print("=" * 80)

    # 创建模型
    print("\n加载 BERT-Large 模型...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-large-uncased',
        num_labels=1,
        local_files_only=True,
        cache_dir='./data/models'
    )
    # 设置为回归任务
    model.config.problem_type = "regression"
    model = model.to(device)

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")
    print(f"可训练参数: {num_trainable:,}")

    # 加载数据
    print("\n加载 STS-B 数据集...")
    train_loader, val_loader = get_stsb_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        max_length=args.max_length,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
    )

    print(f"训练数据批次数: {len(train_loader)}")
    print(f"验证数据批次数: {len(val_loader)}")

    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 开始训练
    print("\n" + "=" * 80)
    train_n_steps(
        model=model,
        train_loader=train_loader,
        eval_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_steps=args.num_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == '__main__':
    main()
