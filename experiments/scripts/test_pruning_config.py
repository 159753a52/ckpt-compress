"""
自定义剪枝配置测试脚本

支持为每个 block 的每个参数组指定不同的剪枝比例，测试剪枝前后的损失变化。

功能:
    - 支持灵活的剪枝配置（JSON 或 Python dict）
    - 测试剪枝前后的损失变化
    - 导出剪枝配置和结果
    - 支持单个 block 或多个 blocks 的配置
    - 支持 GPT-2 和 BERT 模型
    - 支持语言模型、分类和回归任务

使用示例:
    # GPT-2 Small + WikiText-103
    python experiments/scripts/test_pruning_config.py \
        --model gpt2_small \
        --dataset wikitext103 \
        --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
        --block 0 \
        --attn_qkv 0.25 \
        --attn_proj 0.75 \
        --mlp_fc 0.45 \
        --mlp_proj 0.45

    # BERT-LARGE + SST-2 (分类)
    python experiments/scripts/test_pruning_config.py \
        --model bert_large \
        --dataset sst2 \
        --block 0 \
        --attn_qkv 0.25 \
        --attn_proj 0.75 \
        --mlp_fc 0.45 \
        --mlp_proj 0.45 \
        --num_steps 50

    # BERT-LARGE + STS-B (回归)
    python experiments/scripts/test_pruning_config.py \
        --model bert_large \
        --dataset stsb \
        --block 0 \
        --attn_qkv 0.25 \
        --attn_proj 0.75 \
        --mlp_fc 0.45 \
        --mlp_proj 0.45

配置格式 (JSON):
    {
        "blocks": {
            "0": {
                "attn_qkv": 0.25,
                "attn_proj": 0.75,
                "mlp_fc": 0.45,
                "mlp_proj": 0.45,
                "ln_1": 0.25,
                "ln_2": 0.10
            }
        }
    }
"""

import torch
import torch.nn as nn
import json
import argparse
from pathlib import Path
import sys
from datetime import datetime
import pandas as pd
import random
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 设置随机种子以确保结果可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
# 确保 cuDNN 的确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.models.bert import get_bert_large
from src.ckpt_compress.utils.data_loader import (
    get_wikitext103_dataloader,
    get_sst2_loaders,
    get_mnli_loaders,
    get_stsb_loaders,
)
from src.ckpt_compress.methods.adam_prune.importance import compute_importance_scores_first_order


# =============================================================================
# 模型和数据集加载
# =============================================================================

def get_model(model_name: str, pretrained: bool = False, cache_dir: str = './data/models',
              task_type: str = 'lm', dataset_name: str = 'wikitext103'):
    """获取模型

    Args:
        model_name: 模型名称
        pretrained: 是否使用预训练权重
        cache_dir: 模型缓存目录
        task_type: 任务类型 ('lm', 'classification', 'regression')
        dataset_name: 数据集名称（用于确定类别数）
    """
    if model_name == 'gpt2_small':
        return get_gpt2_small(pretrained=pretrained)
    elif model_name == 'bert_large':
        # 对于 BERT，需要根据任务类型使用不同的模型
        from transformers import BertForSequenceClassification, BertConfig
        import os
        os.makedirs(cache_dir, exist_ok=True)

        if task_type == 'lm':
            # 语言建模任务使用 BertForMaskedLM
            return get_bert_large(pretrained=pretrained, local_files_only=True, cache_dir=cache_dir)
        elif task_type in ('classification', 'regression'):
            # 确定标签数量
            if dataset_name == 'sst2':
                num_labels = 2
            elif dataset_name == 'mnli':
                num_labels = 3
            elif dataset_name == 'stsb':
                num_labels = 1
            else:
                num_labels = 2

            # 分类/回归任务使用 BertForSequenceClassification
            if pretrained:
                model = BertForSequenceClassification.from_pretrained(
                    'bert-large-uncased',
                    local_files_only=True,
                    cache_dir=cache_dir,
                    num_labels=num_labels
                )
                # 对于回归任务，需要修改配置
                if task_type == 'regression':
                    model.config.problem_type = "regression"
            else:
                config = BertConfig.from_pretrained(
                    'bert-large-uncased',
                    local_files_only=True,
                    cache_dir=cache_dir
                )
                config.num_labels = num_labels
                if task_type == 'regression':
                    config.problem_type = "regression"
                model = BertForSequenceClassification(config)
            return model
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_data_loader(dataset_name: str, split: str, batch_size: int, seq_length: int = 512):
    """获取数据加载器"""
    if dataset_name == 'wikitext103':
        return get_wikitext103_dataloader(
            split=split,
            batch_size=batch_size,
            seq_length=seq_length,
            num_workers=0
        )
    elif dataset_name == 'sst2':
        train_loader, val_loader = get_sst2_loaders(
            batch_size=batch_size,
            max_length=seq_length,
            num_workers=0
        )
        return train_loader if split == 'train' else val_loader
    elif dataset_name == 'mnli':
        train_loader, val_matched, val_mismatched = get_mnli_loaders(
            batch_size=batch_size,
            max_length=seq_length,
            num_workers=0
        )
        if split == 'train':
            return train_loader
        else:
            # 默认返回 matched 验证集
            return val_matched
    elif dataset_name == 'stsb':
        train_loader, val_loader = get_stsb_loaders(
            batch_size=batch_size,
            max_length=seq_length,
            num_workers=0
        )
        return train_loader if split == 'train' else val_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_task_type(dataset_name: str) -> str:
    """获取任务类型"""
    task_types = {
        'wikitext103': 'lm',
        'sst2': 'classification',
        'mnli': 'classification',
        'stsb': 'regression',
    }
    return task_types.get(dataset_name, 'lm')


def collect_gradients_and_compute_importance(model, data_loader, device, num_steps=100, task_type='lm'):
    """累积梯度并计算重要性得分

    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        num_steps: 累积步数
        task_type: 任务类型 ('lm', 'classification', 'regression')
    """
    model.train()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # 初始化累积字典
    accumulated_gradients = {}
    accumulated_exp_avg_sq = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            accumulated_gradients[name] = torch.zeros_like(param, device='cpu')
            accumulated_exp_avg_sq[name] = torch.zeros_like(param, device='cpu')

    print(f"累积 {num_steps} 步的梯度...")
    data_iter = iter(data_loader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if task_type == 'lm':
            # 语言模型任务
            criterion = nn.CrossEntropyLoss()
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        elif task_type == 'classification':
            # 分类任务
            criterion = nn.CrossEntropyLoss()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
        elif task_type == 'regression':
            # 回归任务
            criterion = nn.MSELoss()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            logits = logits.squeeze(-1)
            loss = criterion(logits, labels.float())
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        loss.backward()

        # 累积梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulated_gradients[name].add_(param.grad.detach().cpu())

        optimizer.step()

        # 累积 exp_avg_sq
        for name, param in model.named_parameters():
            if name in optimizer.state[param]:
                exp_avg_sq = optimizer.state[param]['exp_avg_sq']
                accumulated_exp_avg_sq[name].add_(exp_avg_sq.detach().cpu())

        del input_ids, labels, loss
        if 'logits' in locals():
            del logits
        if 'shift_logits' in locals():
            del shift_logits
        if 'shift_labels' in locals():
            del shift_labels
        if 'outputs' in locals():
            del outputs

        if device == 'cuda' and step % 10 == 0:
            torch.cuda.empty_cache()

        if (step + 1) % 20 == 0:
            print(f"  进度: {step + 1}/{num_steps}")

    # 取平均
    for name in accumulated_gradients:
        accumulated_gradients[name].div_(num_steps)
        accumulated_exp_avg_sq[name].div_(num_steps)

    # 收集权重
    weights = {name: param.detach().cpu() for name, param in model.named_parameters()}

    # 计算重要性得分
    print("计算重要性得分...")
    scores = compute_importance_scores_first_order(
        weights=weights,
        gradients=accumulated_gradients,
        exp_avg_sq=accumulated_exp_avg_sq,
        alpha=0.5
    )

    del accumulated_gradients, accumulated_exp_avg_sq, weights
    if device == 'cuda':
        torch.cuda.empty_cache()

    return scores


def select_layer_params(layer_group, block_id, all_param_names, model_type='gpt2'):
    """选择指定层次的参数名称（只选择 weight 参数）

    Args:
        layer_group: 层类型 (attn_qkv, attn_proj, mlp_fc, mlp_proj, ln_1, ln_2)
        block_id: block ID
        all_param_names: 所有参数名称列表
        model_type: 模型类型 ('gpt2' 或 'bert')
    """
    if model_type == 'gpt2':
        if layer_group == 'attn_qkv':
            return [n for n in all_param_names if f'transformer.h.{block_id}.attn.c_attn.weight' in n]
        elif layer_group == 'attn_proj':
            return [n for n in all_param_names if f'transformer.h.{block_id}.attn.c_proj.weight' in n]
        elif layer_group == 'mlp_fc':
            return [n for n in all_param_names if f'transformer.h.{block_id}.mlp.c_fc.weight' in n]
        elif layer_group == 'mlp_proj':
            return [n for n in all_param_names if f'transformer.h.{block_id}.mlp.c_proj.weight' in n]
        elif layer_group == 'ln_1':
            return [n for n in all_param_names if f'transformer.h.{block_id}.ln_1.weight' in n]
        elif layer_group == 'ln_2':
            return [n for n in all_param_names if f'transformer.h.{block_id}.ln_2.weight' in n]
        else:
            raise ValueError(f"Unknown layer group: {layer_group}")

    elif model_type == 'bert':
        # BERT 参数结构:
        # - query, key, value (合并为 attn_qkv)
        # - output.dense (attn_proj)
        # - intermediate.dense (mlp_fc)
        # - output.dense (mlp_proj)
        # - attention.output.LayerNorm (ln_1)
        # - output.LayerNorm (ln_2)
        if layer_group == 'attn_qkv':
            return [
                n for n in all_param_names
                if f'bert.encoder.layer.{block_id}.attention.self.query.weight' in n or
                   f'bert.encoder.layer.{block_id}.attention.self.key.weight' in n or
                   f'bert.encoder.layer.{block_id}.attention.self.value.weight' in n
            ]
        elif layer_group == 'attn_proj':
            return [n for n in all_param_names if f'bert.encoder.layer.{block_id}.attention.output.dense.weight' in n]
        elif layer_group == 'mlp_fc':
            return [n for n in all_param_names if f'bert.encoder.layer.{block_id}.intermediate.dense.weight' in n]
        elif layer_group == 'mlp_proj':
            return [n for n in all_param_names if f'bert.encoder.layer.{block_id}.output.dense.weight' in n]
        elif layer_group == 'ln_1':
            return [n for n in all_param_names if f'bert.encoder.layer.{block_id}.attention.output.LayerNorm.weight' in n]
        elif layer_group == 'ln_2':
            return [n for n in all_param_names if f'bert.encoder.layer.{block_id}.output.LayerNorm.weight' in n]
        else:
            raise ValueError(f"Unknown layer group: {layer_group}")

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_loss(model, eval_batches, device, task_type='lm'):
    """评估模型在多个批次上的平均损失

    Args:
        model: 模型
        eval_batches: 评估批次列表
        device: 设备
        task_type: 任务类型 ('lm', 'classification', 'regression')
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in eval_batches:
            if task_type == 'lm':
                # 语言模型任务
                criterion = nn.CrossEntropyLoss()
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

            elif task_type == 'classification':
                # 分类任务 (SST-2, MNLI)
                criterion = nn.CrossEntropyLoss()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch['labels'].to(device)

                # BERT 返回 logits
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                loss = criterion(logits, labels)

            elif task_type == 'regression':
                # 回归任务 (STS-B)
                criterion = nn.MSELoss()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch['labels'].to(device)

                # BERT 返回 logits
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # STS-B 的 logits 是单值
                logits = logits.squeeze(-1)
                loss = torch.sqrt(criterion(logits, labels))  # RMSE

            else:
                raise ValueError(f"Unknown task type: {task_type}")

            total_loss += loss.item()

    avg_loss = total_loss / len(eval_batches)
    return avg_loss


def apply_pruning_config(model, scores, pruning_config, all_param_names, device, model_type='gpt2'):
    """
    应用剪枝配置

    Args:
        model: 模型
        scores: 重要性得分字典
        pruning_config: 剪枝配置，格式为 {block_id: {layer_type: sparsity}}
        all_param_names: 所有参数名称列表
        device: 设备
        model_type: 模型类型 ('gpt2' 或 'bert')

    Returns:
        masks: 剪枝掩码字典
        pruning_stats: 剪枝统计信息
    """
    masks = {}
    pruning_stats = []

    param_dict = dict(model.named_parameters())

    for block_id_str, layer_config in pruning_config.items():
        block_id = int(block_id_str)

        print(f"\n应用 Block {block_id} 的剪枝配置:")

        for layer_type, sparsity in layer_config.items():
            # 选择参数
            layer_params = select_layer_params(layer_type, block_id, all_param_names, model_type)

            if len(layer_params) == 0:
                print(f"  警告: {layer_type} 未找到参数")
                continue

            # 获取重要性得分
            layer_scores = {name: scores[name] for name in layer_params}
            flat_scores = torch.cat([s.flatten() for s in layer_scores.values()])

            # 计算阈值
            if flat_scores.numel() > 10_000_000:
                sample_size = 10_000_000
                indices = torch.randperm(flat_scores.numel())[:sample_size]
                sampled_scores = flat_scores[indices]
                threshold = torch.quantile(sampled_scores, sparsity)
                del sampled_scores, indices
            else:
                threshold = torch.quantile(flat_scores, sparsity)

            # 创建剪枝掩码
            for name in layer_params:
                mask = (layer_scores[name] >= threshold).float()
                masks[name] = mask

                # 统计信息
                total_params = layer_scores[name].numel()
                pruned_params = (layer_scores[name] < threshold).sum().item()
                actual_sparsity = pruned_params / total_params

                pruning_stats.append({
                    'block_id': block_id,
                    'layer_type': layer_type,
                    'param_name': name,
                    'target_sparsity': sparsity,
                    'actual_sparsity': actual_sparsity,
                    'total_params': total_params,
                    'pruned_params': pruned_params,
                    'kept_params': total_params - pruned_params
                })

                print(f"  {layer_type}: 目标稀疏度 {sparsity*100:.1f}%, "
                      f"实际稀疏度 {actual_sparsity*100:.1f}%, "
                      f"剪枝 {pruned_params:,}/{total_params:,} 参数")

            del layer_scores, flat_scores

    # 应用剪枝掩码
    print("\n应用剪枝掩码...")
    for name, mask in masks.items():
        param = param_dict[name]
        param.data.mul_(mask.to(device))

    return masks, pruning_stats


def restore_weights(model, original_weights):
    """恢复原始权重"""
    param_dict = dict(model.named_parameters())
    for name, weight in original_weights.items():
        param_dict[name].data.copy_(weight)


def test_pruning_config(checkpoint_path, pruning_config, output_dir, device='cuda',
                       num_steps=100, batch_size=4, seq_length=512, num_eval_batches=10,
                       model_name='gpt2_small', dataset_name='wikitext103'):
    """测试剪枝配置

    Args:
        checkpoint_path: 检查点路径（可选，如果为 None 则使用预训练模型）
        pruning_config: 剪枝配置
        output_dir: 输出目录
        device: 设备
        num_steps: 梯度累积步数
        batch_size: 批次大小
        seq_length: 序列长度
        num_eval_batches: 评估批次数
        model_name: 模型名称 ('gpt2_small' 或 'bert_large')
        dataset_name: 数据集名称 ('wikitext103', 'sst2', 'mnli', 'stsb')
    """
    print("=" * 80)
    print("自定义剪枝配置测试")
    print("=" * 80)
    print(f"模型: {model_name}")
    print(f"数据集: {dataset_name}")
    if checkpoint_path:
        print(f"检查点: {checkpoint_path}")
    else:
        print(f"使用预训练模型")
    print(f"梯度累积步数: {num_steps}")
    print(f"评估批次数: {num_eval_batches}")
    print("=" * 80)

    # 确定模型类型
    model_type = 'bert' if 'bert' in model_name else 'gpt2'
    task_type = get_task_type(dataset_name)

    # 加载模型和数据
    print("\n加载模型...")
    if checkpoint_path:
        model = get_model(model_name, pretrained=False, task_type=task_type, dataset_name=dataset_name)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 使用预训练模型
        model = get_model(model_name, pretrained=True, task_type=task_type, dataset_name=dataset_name)

    model = model.to(device)
    print(f"✓ 模型已加载")

    print("\n加载数据...")
    data_loader = get_data_loader(dataset_name, split='train', batch_size=batch_size, seq_length=seq_length)
    print(f"✓ 数据已加载")

    # 准备评估批次
    print(f"\n准备 {num_eval_batches} 个评估批次...")
    eval_batches = []
    eval_iter = iter(data_loader)
    for i in range(num_eval_batches):
        try:
            batch = next(eval_iter)
            eval_batches.append(batch)
        except StopIteration:
            break
    print(f"✓ 已准备 {len(eval_batches)} 个评估批次")

    # 计算基线损失
    print("\n计算基线损失...")
    baseline_loss = evaluate_loss(model, eval_batches, device, task_type=task_type)
    print(f"✓ 基线损失: {baseline_loss:.6f}")

    # 累积梯度并计算重要性
    print("\n累积梯度并计算重要性得分...")
    scores = collect_gradients_and_compute_importance(
        model, data_loader, device, num_steps=num_steps, task_type=task_type
    )
    print(f"✓ 已计算重要性得分")

    # 保存原始权重
    print("\n保存原始权重...")
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()

    # 应用剪枝配置
    print("\n" + "=" * 80)
    print("应用剪枝配置")
    print("=" * 80)

    all_param_names = list(scores.keys())
    masks, pruning_stats = apply_pruning_config(
        model, scores, pruning_config, all_param_names, device, model_type=model_type
    )

    # 创建统计数据框
    stats_df = pd.DataFrame(pruning_stats)

    # 评估剪枝后的损失
    print("\n计算剪枝后的损失...")
    pruned_loss = evaluate_loss(model, eval_batches, device, task_type=task_type)
    loss_increase = pruned_loss - baseline_loss

    print(f"\n✓ 剪枝后损失: {pruned_loss:.6f}")
    print(f"✓ 损失增量: {loss_increase:+.6f}")

    # 打印统计信息
    print("\n" + "=" * 80)
    print("剪枝统计")
    print("=" * 80)

    # 整体统计
    total_params = stats_df['total_params'].sum()
    total_pruned = stats_df['pruned_params'].sum()
    overall_sparsity = total_pruned / total_params if total_params > 0 else 0

    print("\n【整体统计】")
    print(f"  总参数数:     {total_params:,}")
    print(f"  剪枝参数数:   {total_pruned:,}")
    print(f"  保留参数数:   {total_params - total_pruned:,}")
    print(f"  整体稀疏度:   {overall_sparsity*100:.2f}%")

    # 按层次统计
    print("\n【按层次统计】")
    print(f"{'层次类型':<15} {'总参数':>12} {'剪枝':>12} {'保留':>12} {'稀疏度':>10}")
    print("-" * 65)

    layer_names_map = {
        'attn_qkv': 'Attention QKV',
        'attn_proj': 'Attention Proj',
        'mlp_fc': 'MLP FC',
        'mlp_proj': 'MLP Proj',
        'ln_1': 'LayerNorm 1',
        'ln_2': 'LayerNorm 2'
    }

    for layer_type in ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']:
        layer_stats = stats_df[stats_df['layer_type'] == layer_type]
        if len(layer_stats) > 0:
            layer_total = layer_stats['total_params'].sum()
            layer_pruned = layer_stats['pruned_params'].sum()
            layer_kept = layer_stats['kept_params'].sum()
            layer_sparsity = layer_pruned / layer_total if layer_total > 0 else 0
            layer_name = layer_names_map.get(layer_type, layer_type)

            print(f"{layer_name:<15} {layer_total:>12,} {layer_pruned:>12,} "
                  f"{layer_kept:>12,} {layer_sparsity*100:>9.2f}%")

    # 恢复权重
    print("\n恢复原始权重...")
    restore_weights(model, original_weights)

    # 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存剪枝配置
    config_path = output_dir / 'pruning_config.json'
    with open(config_path, 'w') as f:
        json.dump(pruning_config, f, indent=2)
    print(f"\n✓ 剪枝配置已保存: {config_path}")

    # 保存剪枝统计
    stats_df = pd.DataFrame(pruning_stats)
    stats_path = output_dir / 'pruning_stats.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ 剪枝统计已保存: {stats_path}")

    # 保存结果摘要
    summary = {
        'checkpoint': str(checkpoint_path),
        'baseline_loss': float(baseline_loss),
        'pruned_loss': float(pruned_loss),
        'loss_increase': float(loss_increase),
        'num_eval_batches': num_eval_batches,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pruning_config': pruning_config,
        'total_params_pruned': int(stats_df['pruned_params'].sum()),
        'total_params': int(stats_df['total_params'].sum()),
        'overall_sparsity': float(stats_df['pruned_params'].sum() / stats_df['total_params'].sum())
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ 结果摘要已保存: {summary_path}")

    # 生成报告
    report_path = output_dir / 'REPORT.md'
    generate_report(report_path, summary, stats_df)
    print(f"✓ 报告已保存: {report_path}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)

    return summary, pruning_stats


def generate_report(report_path, summary, stats_df):
    """生成测试报告"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 剪枝配置测试报告\n\n")
        f.write(f"**生成时间**: {summary['timestamp']}\n\n")
        f.write("---\n\n")

        # 损失对比
        f.write("## 损失对比\n\n")
        f.write(f"- **基线损失**: {summary['baseline_loss']:.6f}\n")
        f.write(f"- **剪枝后损失**: {summary['pruned_loss']:.6f}\n")
        f.write(f"- **损失增量**: {summary['loss_increase']:+.6f}\n")

        if summary['loss_increase'] <= 0.001:
            f.write(f"- **结论**: ✅ 无损剪枝（损失增量 ≤ 0.001）\n")
        elif summary['loss_increase'] <= 0.01:
            f.write(f"- **结论**: ⚠️ 轻微损失（损失增量 ≤ 0.01）\n")
        else:
            f.write(f"- **结论**: ❌ 明显损失（损失增量 > 0.01）\n")

        f.write("\n---\n\n")

        # 整体统计
        f.write("## 整体统计\n\n")
        f.write(f"- **总参数数**: {summary['total_params']:,}\n")
        f.write(f"- **剪枝参数数**: {summary['total_params_pruned']:,}\n")
        f.write(f"- **保留参数数**: {summary['total_params'] - summary['total_params_pruned']:,}\n")
        f.write(f"- **整体稀疏度**: {summary['overall_sparsity']*100:.2f}%\n")

        f.write("\n---\n\n")

        # 按层次统计（新增）
        f.write("## 按层次统计\n\n")
        f.write("| 层次类型 | 总参数数 | 剪枝参数数 | 保留参数数 | 稀疏度 |\n")
        f.write("|----------|----------|------------|------------|--------|\n")

        for layer_type in ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2']:
            layer_stats = stats_df[stats_df['layer_type'] == layer_type]
            if len(layer_stats) > 0:
                total_params = layer_stats['total_params'].sum()
                pruned_params = layer_stats['pruned_params'].sum()
                kept_params = layer_stats['kept_params'].sum()
                sparsity = pruned_params / total_params if total_params > 0 else 0

                # 格式化层次名称
                layer_names = {
                    'attn_qkv': 'Attention QKV',
                    'attn_proj': 'Attention Proj',
                    'mlp_fc': 'MLP FC',
                    'mlp_proj': 'MLP Proj',
                    'ln_1': 'LayerNorm 1',
                    'ln_2': 'LayerNorm 2'
                }
                layer_name = layer_names.get(layer_type, layer_type)

                f.write(f"| {layer_name} | {total_params:,} | {pruned_params:,} | "
                       f"{kept_params:,} | {sparsity*100:.2f}% |\n")

        f.write("\n---\n\n")

        # 剪枝配置
        f.write("## 剪枝配置\n\n")
        f.write("```json\n")
        f.write(json.dumps(summary['pruning_config'], indent=2))
        f.write("\n```\n\n")

        f.write("---\n\n")

        # 详细统计
        f.write("## 详细统计\n\n")

        for block_id in sorted(stats_df['block_id'].unique()):
            block_stats = stats_df[stats_df['block_id'] == block_id]
            f.write(f"### Block {block_id}\n\n")
            f.write("| Layer | Target Sparsity | Actual Sparsity | Pruned Params | Total Params |\n")
            f.write("|-------|-----------------|-----------------|---------------|-------------|\n")

            for _, row in block_stats.iterrows():
                f.write(f"| {row['layer_type']} | {row['target_sparsity']*100:.1f}% | "
                       f"{row['actual_sparsity']*100:.1f}% | {row['pruned_params']:,} | "
                       f"{row['total_params']:,} |\n")

            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='测试自定义剪枝配置')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点路径（可选，不指定则使用预训练模型）')
    parser.add_argument('--config', type=str,
                        help='剪枝配置文件路径（JSON格式）')

    # 模型和数据集选择
    parser.add_argument('--model', type=str, default='gpt2_small',
                        choices=['gpt2_small', 'bert_large'],
                        help='模型选择')
    parser.add_argument('--dataset', type=str, default='wikitext103',
                        choices=['wikitext103', 'sst2', 'mnli', 'stsb'],
                        help='数据集选择')

    # 单个 block 的命令行配置
    parser.add_argument('--block', type=int,
                        help='Block ID（用于命令行配置）')
    parser.add_argument('--attn_qkv', type=float,
                        help='Attention QKV 稀疏度')
    parser.add_argument('--attn_proj', type=float,
                        help='Attention Proj 稀疏度')
    parser.add_argument('--mlp_fc', type=float,
                        help='MLP FC 稀疏度')
    parser.add_argument('--mlp_proj', type=float,
                        help='MLP Proj 稀疏度')
    parser.add_argument('--ln_1', type=float,
                        help='LayerNorm 1 稀疏度')
    parser.add_argument('--ln_2', type=float,
                        help='LayerNorm 2 稀疏度')

    # 实验参数
    parser.add_argument('--num_steps', type=int, default=100,
                        help='梯度累积步数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='序列长度')
    parser.add_argument('--num_eval_batches', type=int, default=10,
                        help='评估批次数')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备')
    parser.add_argument('--output_dir', type=str,
                        default='results/pruning_config_test',
                        help='输出目录')

    args = parser.parse_args()

    # 解析剪枝配置
    if args.config:
        # 从文件加载配置
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            pruning_config = config_data.get('blocks', config_data)
    elif args.block is not None:
        # 从命令行参数构建配置
        pruning_config = {
            str(args.block): {}
        }

        if args.attn_qkv is not None:
            pruning_config[str(args.block)]['attn_qkv'] = args.attn_qkv
        if args.attn_proj is not None:
            pruning_config[str(args.block)]['attn_proj'] = args.attn_proj
        if args.mlp_fc is not None:
            pruning_config[str(args.block)]['mlp_fc'] = args.mlp_fc
        if args.mlp_proj is not None:
            pruning_config[str(args.block)]['mlp_proj'] = args.mlp_proj
        if args.ln_1 is not None:
            pruning_config[str(args.block)]['ln_1'] = args.ln_1
        if args.ln_2 is not None:
            pruning_config[str(args.block)]['ln_2'] = args.ln_2
    else:
        print("错误: 必须提供 --config 或 --block 参数")
        return

    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'

    test_pruning_config(
        checkpoint_path=args.checkpoint,
        pruning_config=pruning_config,
        output_dir=args.output_dir,
        device=device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_eval_batches=args.num_eval_batches,
        model_name=args.model,
        dataset_name=args.dataset
    )


if __name__ == '__main__':
    main()
