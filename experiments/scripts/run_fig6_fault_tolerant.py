"""
Fig 6: 容错训练模拟实验

模拟多次从压缩 checkpoint 恢复的训练过程，展示累积误差。

对比方法:
  - none: 不压缩（oracle baseline）
  - magnitude+uniform
  - first-order+uniform (Inshrinkerator-style)
  - second-order+gamma-adaptive (Ours)

运行示例:
    python experiments/scripts/run_fig6_fault_tolerant.py \
        --model gpt2-small --dataset wikitext103 \
        --total_steps 1000 --num_recoveries 5 \
        --prune_ratio 0.5 \
        --device cuda
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lib.models import load_model
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.evaluation import evaluate
from experiments.lib.results import save_results
from src.ckpt_compress.pruning import Pruner, filter_prunable_params


METHODS = [
    'none',
    'magnitude+uniform',
    'first-order+uniform',
    'second-order+gamma-adaptive',
]


def train_one_step(model, optimizer, batch, task_type, device):
    """训练一步，返回 loss。"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    if task_type == 'lm':
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    elif task_type == 'cls':
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attn = batch.get('attention_mask')
        if attn is not None:
            attn = attn.to(device)
        outputs = model(input_ids, attention_mask=attn)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        loss = criterion(logits, labels)
    elif task_type == 'cv':
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(images)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        loss = criterion(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def collect_gradients_inline(model, optimizer, train_loader, device, num_steps, task_type):
    """在训练过程中收集梯度和 Adam 状态（不创建新优化器）。"""
    criterion = nn.CrossEntropyLoss()
    accumulated_grads = defaultdict(lambda: 0)
    accumulated_eas = defaultdict(lambda: 0)
    
    data_iter = iter(train_loader)
    model.train()
    
    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        optimizer.zero_grad()
        
        if task_type == 'lm':
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        elif task_type == 'cls':
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attn = batch.get('attention_mask')
            if attn is not None:
                attn = attn.to(device)
            outputs = model(input_ids, attention_mask=attn)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
        elif task_type == 'cv':
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
        
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulated_grads[name] = accumulated_grads[name] + param.grad.detach().cpu()
        
        optimizer.step()
        
        for name, param in model.named_parameters():
            if param in optimizer.state and 'exp_avg_sq' in optimizer.state[param]:
                eas = optimizer.state[param]['exp_avg_sq']
                accumulated_eas[name] = accumulated_eas[name] + eas.detach().cpu()
    
    for name in accumulated_grads:
        accumulated_grads[name] /= num_steps
    for name in accumulated_eas:
        accumulated_eas[name] /= num_steps
    
    weights = {name: p.detach().cpu() for name, p in model.named_parameters()}
    return weights, dict(accumulated_grads), dict(accumulated_eas)


def apply_pruning_to_model_and_optimizer(model, optimizer, scores, layer_ratios, device):
    """对模型和优化器状态同时应用剪枝掩码。"""
    model_params = dict(model.named_parameters())
    
    for name, score in scores.items():
        if name not in model_params or name not in layer_ratios:
            continue
        
        ratio = layer_ratios[name]
        flat = score.flatten()
        k = int(len(flat) * ratio)
        if k == 0:
            continue
        
        threshold = torch.kthvalue(flat, k).values
        mask = (score > threshold).float().to(device)
        
        param = model_params[name]
        param.data.mul_(mask)
        
        if param in optimizer.state:
            if 'exp_avg' in optimizer.state[param]:
                optimizer.state[param]['exp_avg'].mul_(mask)
            if 'exp_avg_sq' in optimizer.state[param]:
                optimizer.state[param]['exp_avg_sq'].mul_(mask)


def run_method(method_name, args, train_loader, val_batches, task_type):
    """运行单个方法的完整训练流程。"""
    print(f"\n{'='*60}")
    print(f"方法: {method_name}")
    print(f"{'='*60}")
    
    model, _ = load_model(args.model, pretrained=True, 
                           checkpoint_path=args.checkpoint, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    steps_per_segment = args.total_steps // (args.num_recoveries + 1)
    loss_history = []
    global_step = 0
    data_iter = iter(train_loader)
    
    for seg in range(args.num_recoveries + 1):
        print(f"\n--- Segment {seg+1}/{args.num_recoveries+1} ---")
        
        for step in range(steps_per_segment):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            loss = train_one_step(model, optimizer, batch, task_type, args.device)
            global_step += 1
            
            if step % args.eval_interval == 0:
                metrics = evaluate(model, val_batches, task_type, args.device)
                ppl = metrics.get('perplexity', 0)
                acc = metrics.get('accuracy', 0)
                loss_history.append({
                    'step': global_step, 'loss': metrics['loss'],
                    'perplexity': ppl, 'accuracy': acc,
                })
        
        # 在恢复点压缩（最后一段不压缩）
        if seg < args.num_recoveries and method_name != 'none':
            print(f"  [恢复点 {seg+1}] 压缩 checkpoint...")
            
            parts = method_name.split('+')
            importance, allocation = parts[0], parts[1]
            
            weights, grads, eas = collect_gradients_inline(
                model, optimizer, train_loader, args.device, 
                args.num_importance_steps, task_type)
            
            prunable_w = filter_prunable_params(weights)
            prunable_g = {k: grads[k] for k in prunable_w if k in grads}
            prunable_e = {k: eas[k] for k in prunable_w if k in eas}
            
            pruner = Pruner(importance=importance, allocation=allocation, alpha=args.alpha)
            scores = pruner.compute_scores(prunable_w, prunable_g, prunable_e)
            layer_ratios = pruner.compute_layer_ratios(scores, args.prune_ratio)
            
            apply_pruning_to_model_and_optimizer(model, optimizer, scores, layer_ratios, args.device)
            
            metrics = evaluate(model, val_batches, task_type, args.device)
            print(f"  剪枝后: loss={metrics['loss']:.4f}")
    
    return loss_history


def plot_results(all_results, args):
    """绘制 loss 曲线。"""
    plt.figure(figsize=(12, 6))
    colors = {'none': 'black', 'magnitude+uniform': 'blue',
              'first-order+uniform': 'green', 'second-order+gamma-adaptive': 'red'}
    
    for method, history in all_results.items():
        steps = [h['step'] for h in history]
        losses = [h['loss'] for h in history]
        plt.plot(steps, losses, label=method, color=colors.get(method, 'gray'), linewidth=2)
    
    steps_per_seg = args.total_steps // (args.num_recoveries + 1)
    for i in range(1, args.num_recoveries + 1):
        plt.axvline(x=i * steps_per_seg, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Validation Loss')
    plt.title(f'Fault-Tolerant Training: {args.model} on {args.dataset}\n'
              f'({args.num_recoveries} recoveries, prune_ratio={args.prune_ratio})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f'fig6_{args.model}_{args.dataset}.png', dpi=300)
    plt.savefig(out_dir / f'fig6_{args.model}_{args.dataset}.pdf')
    plt.close()
    print(f"图片已保存: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Fig 6: 容错训练模拟')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--total_steps', type=int, default=1000)
    parser.add_argument('--num_recoveries', type=int, default=5)
    parser.add_argument('--prune_ratio', type=float, default=0.5)
    parser.add_argument('--methods', type=str, default=None,
                        help='逗号分隔的方法列表，默认全部')
    parser.add_argument('--num_importance_steps', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/paper_results/fig6')
    args = parser.parse_args()
    
    methods = [m.strip() for m in args.methods.split(',')] if args.methods else METHODS
    
    print("=" * 60)
    print("Fig 6: 容错训练模拟实验")
    print(f"模型: {args.model} | 数据集: {args.dataset}")
    print(f"总步数: {args.total_steps} | 恢复次数: {args.num_recoveries}")
    print(f"剪枝率: {args.prune_ratio} | 方法: {methods}")
    print("=" * 60)
    
    train_loader, val_loader, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length)
    
    val_batches = cache_batches(val_loader, 10, task_type)
    
    all_results = {}
    for method in methods:
        history = run_method(method, args, train_loader, val_batches, task_type)
        all_results[method] = history
    
    # 保存
    flat_results = []
    for method, history in all_results.items():
        for h in history:
            flat_results.append({'method': method, **h})
    
    save_results(flat_results, args.output_dir, f"fig6_{args.model}_{args.dataset}",
                 config=vars(args))
    plot_results(all_results, args)
    print("\n实验完成！")


if __name__ == '__main__':
    main()
