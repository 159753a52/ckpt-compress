"""
BERT-Large 细粒度剪枝分析 - 只关注 weight 参数

分析 BERT-Large 所有 24 个 blocks 中每个 weight 参数的无损剪枝比例。

细粒度层次划分:
    Attention:
        - attn_q: query.weight [1024, 1024]
        - attn_k: key.weight [1024, 1024]
        - attn_v: value.weight [1024, 1024]
        - attn_out: output.dense.weight [1024, 1024]
    MLP (FFN):
        - ffn_fc: intermediate.dense.weight [4096, 1024]
        - ffn_proj: output.dense.weight [1024, 4096]
    LayerNorm:
        - ln_attn: attention.output.LayerNorm.weight [1024]
        - ln_ffn: output.LayerNorm.weight [1024]

运行命令:
    # SST-2 (分类任务 - 使用准确率)
    python experiments/scripts/bert_fine_grained_pruning_analysis.py \
        --dataset sst2 \
        --num_steps 50 \
        --batch_size 16 \
        --num_eval_batches 10 \
        --device cuda \
        --output_dir results/bert_fine_grained_pruning_sst2

    # MNLI (分类任务 - 使用准确率)
    python experiments/scripts/bert_fine_grained_pruning_analysis.py \
        --dataset mnli \
        --num_steps 50 \
        --batch_size 16 \
        --num_eval_batches 10

    # STS-B (回归任务 - 使用皮尔逊相关系数)
    python experiments/scripts/bert_fine_gruned_pruning_analysis.py \
        --dataset stsb \
        --num_steps 50 \
        --batch_size 16 \
        --num_eval_batches 10

实验设计:
    - 测试范围: 所有 24 个 Transformer Blocks + Embedding
    - 参数组: attn_q, attn_k, attn_v, attn_out, ffn_fc, ffn_proj, ln_attn, ln_ffn (每个 block)
    - 稀疏度: 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%
    - 重要性度量: |g_i * θ_i|
    - 无损标准: 准确率下降 <= 0.5% 或相关系数下降 <= 0.01
    - 只测试 weight 参数（排除 bias）
"""

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm
import argparse
from datetime import datetime
import gc
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.bert import get_bert_large
from src.ckpt_compress.utils.data_loader import (
    get_sst2_loaders,
    get_mnli_loaders,
    get_stsb_loaders,
)
from src.ckpt_compress.methods.adam_prune.importance import compute_importance_scores_first_order
from transformers import BertForSequenceClassification


def get_model_and_task_type(dataset_name: str, checkpoint_path: Optional[str] = None):
    """根据数据集获取模型和任务类型

    Args:
        dataset_name: 数据集名称
        checkpoint_path: 检查点路径（如果为 None，使用默认的 1000 步检查点）
    """
    from transformers import BertForSequenceClassification

    # 确定标签数量
    if dataset_name == 'sst2':
        num_labels = 2
        default_checkpoint = 'checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000_final.pt'
    elif dataset_name == 'mnli':
        num_labels = 3
        default_checkpoint = 'checkpoints/bert_large_mnli_1000steps/checkpoint_step_1000_final.pt'
    elif dataset_name == 'stsb':
        num_labels = 1
        default_checkpoint = 'checkpoints/bert_large_stsb_1000steps/checkpoint_step_1000_final.pt'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 使用默认检查点路径（如果没有指定）
    if checkpoint_path is None:
        checkpoint_path = default_checkpoint

    # 检查检查点文件是否存在
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f"警告: 检查点文件不存在: {checkpoint_path}")
        print(f"使用预训练模型代替...")
        checkpoint_path = None

    # 加载模型
    if checkpoint_path is not None:
        print(f"从检查点加载模型: {checkpoint_path}")
        # 先创建模型结构
        model = BertForSequenceClassification.from_pretrained(
            'bert-large-uncased',
            local_files_only=True,
            cache_dir='./data/models',
            num_labels=num_labels
        )
        # 加载检查点权重
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 已从检查点加载模型权重 (step={checkpoint.get('step', 'unknown')})")
    else:
        # 使用预训练模型
        model = BertForSequenceClassification.from_pretrained(
            'bert-large-uncased',
            local_files_only=True,
            cache_dir='./data/models',
            num_labels=num_labels
        )

    # 对于回归任务，设置 problem_type
    if dataset_name == 'stsb':
        model.config.problem_type = "regression"

    # 确定任务类型
    if dataset_name == 'stsb':
        task_type = 'regression'
    else:
        task_type = 'classification'

    return model, task_type


def get_data_loader(dataset_name: str, batch_size: int = 16):
    """获取数据加载器"""
    if dataset_name == 'sst2':
        train_loader, _ = get_sst2_loaders(batch_size=batch_size, num_workers=0)
        return train_loader
    elif dataset_name == 'mnli':
        train_loader, _, _ = get_mnli_loaders(batch_size=batch_size, num_workers=0)
        return train_loader
    elif dataset_name == 'stsb':
        train_loader, _ = get_stsb_loaders(batch_size=batch_size, num_workers=0)
        return train_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def collect_gradients_and_compute_importance(model, data_loader, device, num_steps=100, task_type='classification'):
    """累积梯度并计算重要性得分（内存优化版本）

    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        num_steps: 累积步数
        task_type: 任务类型 ('classification' 或 'regression')
    """
    model.train()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # 初始化累积字典
    accumulated_gradients = {}
    accumulated_exp_avg_sq = {}
    params_without_grad = []  # 记录没有梯度的参数

    for name, param in model.named_parameters():
        if param.requires_grad:
            accumulated_gradients[name] = torch.zeros_like(param, device='cpu')
            accumulated_exp_avg_sq[name] = torch.zeros_like(param, device='cpu')

    print(f"累积 {num_steps} 步的梯度...")
    data_iter = iter(data_loader)

    for step in tqdm(range(num_steps)):
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

        if task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
        elif task_type == 'regression':
            criterion = nn.MSELoss()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            logits = logits.squeeze(-1)
            loss = criterion(logits, labels.float())
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        loss.backward()

        # 累积梯度（添加调试信息）
        for name, param in model.named_parameters():
            if param.grad is not None:
                accumulated_gradients[name].add_(param.grad.detach().cpu())
            else:
                if step == 0 and name not in params_without_grad:
                    params_without_grad.append(name)

        optimizer.step()

        # 累积 exp_avg_sq
        for name, param in model.named_parameters():
            if name in optimizer.state[param]:
                exp_avg_sq = optimizer.state[param]['exp_avg_sq']
                accumulated_exp_avg_sq[name].add_(exp_avg_sq.detach().cpu())

        # 释放中间变量
        del input_ids, labels, loss
        if 'logits' in locals():
            del logits
        if 'outputs' in locals():
            del outputs

        # 定期清理GPU缓存
        if device == 'cuda' and step % 10 == 0:
            torch.cuda.empty_cache()

    # 警告：没有梯度的参数
    if params_without_grad:
        print(f"\n⚠️  警告: 以下 {len(params_without_grad)} 个参数在第一步没有梯度:")
        for name in params_without_grad[:5]:  # 只显示前5个
            print(f"  - {name}")
        if len(params_without_grad) > 5:
            print(f"  ... 还有 {len(params_without_grad) - 5} 个参数")

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

    # 调试：检查得分统计
    print("\n重要性得分统计:")
    zero_score_params = []
    for name, score in scores.items():
        if (score == 0).all():
            zero_score_params.append(name)

    if zero_score_params:
        print(f"⚠️  警告: 以下 {len(zero_score_params)} 个参数的重要性得分全为 0:")
        for name in zero_score_params[:5]:
            print(f"  - {name}")
        if len(zero_score_params) > 5:
            print(f"  ... 还有 {len(zero_score_params) - 5} 个参数")
    else:
        print("✓ 所有参数都有非零重要性得分")

    # 清理内存
    del accumulated_gradients, accumulated_exp_avg_sq, weights
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return scores


def select_layer_params(layer_group, block_id, all_param_names):
    """
    选择指定层次的参数名称（只选择 weight 参数）
    """
    if layer_group == 'embedding':
        return [n for n in all_param_names if 'bert.embeddings' in n and 'weight' in n]

    elif layer_group == 'attn_q':
        return [n for n in all_param_names
                if f'bert.encoder.layer.{block_id}.attention.self.query.weight' in n]

    elif layer_group == 'attn_k':
        return [n for n in all_param_names
                if f'bert.encoder.layer.{block_id}.attention.self.key.weight' in n]

    elif layer_group == 'attn_v':
        return [n for n in all_param_names
                if f'bert.encoder.layer.{block_id}.attention.self.value.weight' in n]

    elif layer_group == 'attn_out':
        return [n for n in all_param_names
                if f'bert.encoder.layer.{block_id}.attention.output.dense.weight' in n]

    elif layer_group == 'ffn_fc':
        return [n for n in all_param_names
                if f'bert.encoder.layer.{block_id}.intermediate.dense.weight' in n]

    elif layer_group == 'ffn_proj':
        return [n for n in all_param_names
                if f'bert.encoder.layer.{block_id}.output.dense.weight' in n]

    elif layer_group == 'ln_attn':
        return [n for n in all_param_names
                if f'bert.encoder.layer.{block_id}.attention.output.LayerNorm.weight' in n]

    elif layer_group == 'ln_ffn':
        return [n for n in all_param_names
                if f'bert.encoder.layer.{block_id}.output.LayerNorm.weight' in n]

    else:
        raise ValueError(f"Unknown layer group: {layer_group}")


def evaluate_loss(model, eval_batches, device, task_type='classification'):
    """评估模型在多个批次上的平均损失

    Args:
        model: 模型
        eval_batches: 评估批次列表
        device: 设备
        task_type: 任务类型

    Returns:
        float: 平均损失值
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in eval_batches:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'].to(device)

            if task_type == 'classification':
                criterion = nn.CrossEntropyLoss()
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = criterion(logits, labels)
            elif task_type == 'regression':
                criterion = nn.MSELoss()
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                logits = logits.squeeze(-1)
                loss = criterion(logits, labels.float())
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            total_loss += loss.item()

    avg_loss = total_loss / len(eval_batches)
    return avg_loss


def prune_and_evaluate(model, layer_params, scores, sparsity, eval_batches, device, task_type='classification', debug=False):
    """剪枝指定层次并评估损失"""
    # 获取该层次的重要性得分
    layer_scores = {name: scores[name] for name in layer_params}

    # 调试：检查重要性得分分布
    if debug:
        print(f"\n  === 调试：重要性得分分布 ===")
        for name in layer_params:
            score = layer_scores[name]
            print(f"  {name}:")
            print(f"    min={score.min():.6e}, max={score.max():.6e}, mean={score.mean():.6e}")
            print(f"    零值数量: {(score == 0).sum().item()}/{score.numel()}")

    # 展平得分
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

    if debug:
        print(f"  阈值 (sparsity={sparsity*100:.1f}%): {threshold:.6e}")
        print(f"  得分范围: [{flat_scores.min():.6e}, {flat_scores.max():.6e}]")

        # 检查是否所有得分都相同
        if flat_scores.min() == flat_scores.max():
            print(f"  ⚠️  警告: 所有重要性得分都相同！无法有效剪枝")

    # 创建剪枝掩码
    masks = {}
    for name in layer_params:
        mask = (layer_scores[name] >= threshold).float()
        masks[name] = mask

    # 保存原始权重
    original_weights = {}
    param_dict = dict(model.named_parameters())
    for name in layer_params:
        param = param_dict[name]
        original_weights[name] = param.data.clone()

    # 应用剪枝（添加详细调试）
    if debug:
        print(f"\n  === 调试：剪枝操作 ===")

    for name in layer_params:
        param = param_dict[name]
        mask = masks[name].to(device)

        if debug:
            # 剪枝前
            before_zeros = (param.data == 0).sum().item()
            before_nonzeros = (param.data != 0).sum().item()
            before_mean = param.data.abs().mean().item()

        # 应用剪枝
        param.data.mul_(mask)

        if debug:
            # 剪枝后
            after_zeros = (param.data == 0).sum().item()
            after_nonzeros = (param.data != 0).sum().item()
            after_mean = param.data.abs().mean().item()

            print(f"  {name}:")
            print(f"    剪枝前: {before_zeros:,} 个零, {before_nonzeros:,} 个非零, |mean|={before_mean:.6e}")
            print(f"    剪枝后: {after_zeros:,} 个零, {after_nonzeros:,} 个非零, |mean|={after_mean:.6e}")
            print(f"    新增零: {after_zeros - before_zeros:,} ({(after_zeros - before_zeros)/param.data.numel()*100:.2f}%)")

    # 调试信息：检查实际剪枝比例
    total_pruned = 0
    total_params = 0
    for name in layer_params:
        mask = masks[name]
        total_pruned += (mask == 0).sum().item()
        total_params += mask.numel()
    actual_sparsity = total_pruned / total_params if total_params > 0 else 0

    if debug:
        print(f"  总体实际剪枝比例: {actual_sparsity*100:.2f}%")

    # 评估损失（改为评估损失而不是准确率）
    if debug:
        print(f"\n  === 调试：评估模型 ===")
        print(f"  模型模式: {'train' if model.training else 'eval'}")
        print(f"  评估批次数: {len(eval_batches)}")

    avg_loss = evaluate_loss(model, eval_batches, device, task_type=task_type)

    if debug:
        print(f"  评估结果（损失）: {avg_loss:.4f}")

    # 恢复权重
    for name in layer_params:
        param = param_dict[name]
        param.data.copy_(original_weights[name])

    if debug:
        print(f"\n  === 调试：权重已恢复 ===")
        for name in layer_params:
            param = param_dict[name]
            restored_zeros = (param.data == 0).sum().item()
            print(f"  {name}: {restored_zeros:,} 个零（应该恢复到原始状态）")

    # 统计剪枝信息
    pruned_params = (flat_scores < threshold).sum().item()
    total_params_count = flat_scores.numel()

    # 清理内存
    del layer_scores, flat_scores, masks, original_weights
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'loss': avg_loss,
        'pruned_params': pruned_params,
        'total_params': total_params_count,
        'actual_sparsity': actual_sparsity,
    }


def find_max_sparsity(results_df, layer_name, block_id, tolerance=0.001):
    """找到指定容忍度下的最大剪枝稀疏度

    Args:
        results_df: 结果 DataFrame
        layer_name: 层名称
        block_id: block ID
        tolerance: 容忍阈值（损失增量 <= tolerance）

    Returns:
        float: 最大允许剪枝稀疏度
    """
    layer_results = results_df[
        (results_df['layer'] == layer_name) &
        (results_df['block_id'] == block_id)
    ].sort_values('sparsity')

    if len(layer_results) == 0:
        return 0.0

    # 找到满足容忍度要求的最大稀疏度（损失增量 <= tolerance）
    acceptable_results = layer_results[layer_results['loss_increase'] <= tolerance]

    if len(acceptable_results) == 0:
        return 0.0
    else:
        return acceptable_results['sparsity'].max()


def save_results(results, output_dir, baseline_loss, num_blocks=24, task_type='classification'):
    """保存结果为 CSV 和图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tolerance = 0.001  # 损失增量容忍阈值

    # 保存完整 CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / 'fine_grained_pruning_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 完整CSV已保存: {csv_path}")

    # 计算每个层次的最大允许剪枝比例
    max_sparsity_summary = []

    # Embedding层
    emb_max_sparsity = find_max_sparsity(df, 'embedding', -1, tolerance)
    emb_data = df[(df['layer'] == 'embedding') & (df['block_id'] == -1)]
    emb_total_params = emb_data['total_params'].iloc[0] if len(emb_data) > 0 else 0
    max_sparsity_summary.append({
        'layer': 'Embedding',
        'block_id': -1,
        'max_allowed_sparsity': emb_max_sparsity,
        'max_allowed_percent': emb_max_sparsity * 100,
        'baseline_loss': baseline_loss,
        'total_params': emb_total_params
    })

    # 所有blocks的细粒度层次
    layer_types = ['attn_q', 'attn_k', 'attn_v', 'attn_out', 'ffn_fc', 'ffn_proj', 'ln_attn', 'ln_ffn']
    for block_id in range(num_blocks):
        for layer_type in layer_types:
            max_sparsity = find_max_sparsity(df, layer_type, block_id, tolerance)
            layer_data = df[(df['layer'] == layer_type) & (df['block_id'] == block_id)]
            total_params = layer_data['total_params'].iloc[0] if len(layer_data) > 0 else 0
            max_sparsity_summary.append({
                'layer': layer_type,
                'block_id': block_id,
                'max_allowed_sparsity': max_sparsity,
                'max_allowed_percent': max_sparsity * 100,
                'baseline_loss': baseline_loss,
                'total_params': total_params
            })

    # 保存剪枝摘要
    summary_df = pd.DataFrame(max_sparsity_summary)
    summary_csv_path = output_dir / 'max_allowed_sparsity_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✓ 最大允许剪枝摘要已保存: {summary_csv_path}")

    # 绘制热力图
    plot_sparsity_heatmap(summary_df, output_dir, num_blocks, task_type)

    # 生成分析报告
    generate_analysis_report(summary_df, output_dir, baseline_loss, num_blocks)

    # 导出参数数量统计
    export_parameter_counts(summary_df, output_dir, num_blocks)


def plot_sparsity_heatmap(summary_df, output_dir, num_blocks=24, task_type='classification'):
    """绘制最大允许剪枝比例热力图"""
    blocks_df = summary_df[summary_df['block_id'] >= 0]
    pivot_data = blocks_df.pivot(index='block_id', columns='layer', values='max_allowed_percent')

    # 按照逻辑顺序排列列
    column_order = ['attn_q', 'attn_k', 'attn_v', 'attn_out', 'ffn_fc', 'ffn_proj', 'ln_attn', 'ln_ffn']
    pivot_data = pivot_data[[col for col in column_order if col in pivot_data.columns]]

    fig, ax = plt.subplots(figsize=(14, 10))

    metric_label = "Max Allowed Sparsity (%)"

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        cbar_kws={'label': metric_label},
        vmin=0,
        vmax=90,
        ax=ax
    )

    ax.set_xlabel('Layer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Block ID', fontsize=12, fontweight='bold')
    title_str = f"Maximum Allowed Sparsity by Block and Layer Type (BERT-Large, {num_blocks} Blocks)"
    title_str += "\n(Metric: Loss Increase, Tolerance: 0.001)"
    ax.set_title(title_str, fontsize=14, fontweight='bold')

    # 更友好的列标签
    ax.set_xticklabels(['Attn Q', 'Attn K', 'Attn V', 'Attn Out', 'FFN FC', 'FFN Proj', 'LN Attn', 'LN FFN'])

    plt.tight_layout()
    heatmap_path = output_dir / 'max_allowed_sparsity_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ 热力图已保存: {heatmap_path}")
    plt.close()


def generate_analysis_report(summary_df, output_dir, baseline_loss, num_blocks=24):
    """生成分析报告"""
    report_path = output_dir / 'ANALYSIS_REPORT.md'

    tolerance = 0.001  # 损失增量容忍阈值

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# BERT-Large 细粒度剪枝分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**基线损失**: {baseline_loss:.4f}\n\n")
        f.write(f"**Block 数量**: {num_blocks}\n\n")
        f.write(f"**容忍阈值**: 损失增量 <= {tolerance}\n\n")
        f.write("**分析粒度**: 只测试 weight 参数（排除 bias）\n\n")
        f.write("---\n\n")

        # Embedding层
        f.write("## Embedding Layer\n\n")
        emb_data = summary_df[summary_df['block_id'] == -1]
        if len(emb_data) > 0:
            emb_sparsity = emb_data['max_allowed_percent'].values[0]
            f.write(f"**最大允许剪枝比例**: {emb_sparsity:.1f}%\n\n")

        f.write("---\n\n")

        # 各个block的统计
        f.write(f"## Transformer Blocks (0-{num_blocks-1})\n\n")

        blocks_df = summary_df[summary_df['block_id'] >= 0]

        layer_configs = [
            ('attn_q', 'Attention Query (self.query.weight)'),
            ('attn_k', 'Attention Key (self.key.weight)'),
            ('attn_v', 'Attention Value (self.value.weight)'),
            ('attn_out', 'Attention Output (output.dense.weight)'),
            ('ffn_fc', 'FFN Intermediate (intermediate.dense.weight)'),
            ('ffn_proj', 'FFN Output (output.dense.weight)'),
            ('ln_attn', 'LayerNorm after Attention (output.LayerNorm.weight)'),
            ('ln_ffn', 'LayerNorm after FFN (output.LayerNorm.weight)'),
        ]

        for layer_type, layer_name in layer_configs:
            layer_data = blocks_df[blocks_df['layer'] == layer_type]

            f.write(f"### {layer_name}\n\n")

            if len(layer_data) > 0:
                mean_sparsity = layer_data['max_allowed_percent'].mean()
                std_sparsity = layer_data['max_allowed_percent'].std()
                min_sparsity = layer_data['max_allowed_percent'].min()
                max_sparsity = layer_data['max_allowed_percent'].max()

                f.write(f"**统计信息**:\n")
                f.write(f"- 平均: {mean_sparsity:.1f}%\n")
                f.write(f"- 标准差: {std_sparsity:.1f}%\n")
                f.write(f"- 最小: {min_sparsity:.1f}%\n")
                f.write(f"- 最大: {max_sparsity:.1f}%\n\n")

                # 显示最耐剪枝和最敏感的 blocks
                top5 = layer_data.nlargest(5, 'max_allowed_percent')
                bottom5 = layer_data.nsmallest(5, 'max_allowed_percent')

                f.write("**Top 5 Blocks** (最耐剪枝):\n")
                for _, row in top5.iterrows():
                    f.write(f"- Block {int(row['block_id'])}: {row['max_allowed_percent']:.1f}%\n")
                f.write("\n")

                f.write("**Bottom 5 Blocks** (最敏感):\n")
                for _, row in bottom5.iterrows():
                    f.write(f"- Block {int(row['block_id'])}: {row['max_allowed_percent']:.1f}%\n")
                f.write("\n")

        f.write("---\n\n")

        # 对比分析
        f.write("## 层次对比分析\n\n")

        # Attention: Q vs K vs V vs Out
        f.write("### Attention 层对比\n\n")
        attn_q_data = blocks_df[blocks_df['layer'] == 'attn_q']
        attn_k_data = blocks_df[blocks_df['layer'] == 'attn_k']
        attn_v_data = blocks_df[blocks_df['layer'] == 'attn_v']
        attn_out_data = blocks_df[blocks_df['layer'] == 'attn_out']
        f.write(f"- **Query 平均允许剪枝**: {attn_q_data['max_allowed_percent'].mean():.1f}%\n")
        f.write(f"- **Key 平均允许剪枝**: {attn_k_data['max_allowed_percent'].mean():.1f}%\n")
        f.write(f"- **Value 平均允许剪枝**: {attn_v_data['max_allowed_percent'].mean():.1f}%\n")
        f.write(f"- **Output 平均允许剪枝**: {attn_out_data['max_allowed_percent'].mean():.1f}%\n\n")

        # FFN: FC vs Proj
        f.write("### FFN 层对比\n\n")
        ffn_fc_data = blocks_df[blocks_df['layer'] == 'ffn_fc']
        ffn_proj_data = blocks_df[blocks_df['layer'] == 'ffn_proj']
        f.write(f"- **FC 平均允许剪枝**: {ffn_fc_data['max_allowed_percent'].mean():.1f}%\n")
        f.write(f"- **Proj 平均允许剪枝**: {ffn_proj_data['max_allowed_percent'].mean():.1f}%\n\n")

    print(f"✓ 分析报告已保存: {report_path}")


def export_parameter_counts(summary_df, output_dir, num_blocks=24):
    """导出参数数量统计到 CSV"""
    output_dir = Path(output_dir)

    # 创建参数数量统计表
    param_counts = []

    # Embedding层
    emb_data = summary_df[summary_df['block_id'] == -1]
    if len(emb_data) > 0:
        param_counts.append({
            'layer': 'Embedding',
            'block_id': -1,
            'total_params': int(emb_data['total_params'].iloc[0]),
            'max_allowed_sparsity': emb_data['max_allowed_sparsity'].iloc[0],
            'prunable_params': int(emb_data['total_params'].iloc[0] * emb_data['max_allowed_sparsity'].iloc[0]),
            'remaining_params': int(emb_data['total_params'].iloc[0] * (1 - emb_data['max_allowed_sparsity'].iloc[0]))
        })

    # 所有blocks的细粒度层次
    blocks_df = summary_df[summary_df['block_id'] >= 0]
    layer_types = ['attn_q', 'attn_k', 'attn_v', 'attn_out', 'ffn_fc', 'ffn_proj', 'ln_attn', 'ln_ffn']

    for block_id in range(num_blocks):
        for layer_type in layer_types:
            layer_data = blocks_df[(blocks_df['layer'] == layer_type) & (blocks_df['block_id'] == block_id)]
            if len(layer_data) > 0:
                total = int(layer_data['total_params'].iloc[0])
                sparsity = layer_data['max_allowed_sparsity'].iloc[0]
                param_counts.append({
                    'layer': layer_type,
                    'block_id': int(block_id),
                    'total_params': total,
                    'max_allowed_sparsity': sparsity,
                    'prunable_params': int(total * sparsity),
                    'remaining_params': int(total * (1 - sparsity))
                })

    # 保存为 CSV
    param_df = pd.DataFrame(param_counts)
    param_csv_path = output_dir / 'parameter_counts.csv'
    param_df.to_csv(param_csv_path, index=False)
    print(f"✓ 参数数量统计已保存: {param_csv_path}")

    # 计算并保存汇总统计
    summary_stats = []

    # 按层次类型汇总
    for layer_type in layer_types:
        layer_data = param_df[param_df['layer'] == layer_type]
        if len(layer_data) > 0:
            summary_stats.append({
                'layer_type': layer_type,
                'total_params': int(layer_data['total_params'].sum()),
                'total_prunable': int(layer_data['prunable_params'].sum()),
                'total_remaining': int(layer_data['remaining_params'].sum()),
                'avg_sparsity': layer_data['max_allowed_sparsity'].mean()
            })

    # 总体统计
    blocks_data = param_df[param_df['block_id'] >= 0]
    summary_stats.append({
        'layer_type': 'ALL_BLOCKS',
        'total_params': int(blocks_data['total_params'].sum()),
        'total_prunable': int(blocks_data['prunable_params'].sum()),
        'total_remaining': int(blocks_data['remaining_params'].sum()),
        'avg_sparsity': blocks_data['max_allowed_sparsity'].mean()
    })

    summary_df2 = pd.DataFrame(summary_stats)
    summary_csv_path = output_dir / 'parameter_counts_summary.csv'
    summary_df2.to_csv(summary_csv_path, index=False)
    print(f"✓ 参数数量汇总已保存: {summary_csv_path}")


def run_fine_grained_pruning_experiment(dataset_name, output_dir, device='cuda',
                                        num_steps=50, batch_size=16, num_eval_batches=10,
                                        checkpoint_path=None):
    """运行细粒度剪枝实验

    Args:
        dataset_name: 数据集名称 ('sst2', 'mnli', 'stsb')
        output_dir: 输出目录
        device: 设备
        num_steps: 梯度累积步数
        batch_size: 批次大小
        num_eval_batches: 评估批次数
        checkpoint_path: 检查点路径（默认使用 1000 步检查点）
    """
    num_blocks = 24  # BERT-Large 有 24 个 blocks

    print("=" * 80)
    print(f"BERT-Large 细粒度剪枝分析（只测试 weight 参数）")
    print(f"数据集: {dataset_name.upper()}")
    print("=" * 80)
    print(f"梯度累积步数: {num_steps}")
    print(f"评估批次数: {num_eval_batches}")
    print("=" * 80)

    # 确定任务类型
    task_type = 'regression' if dataset_name == 'stsb' else 'classification'

    # 加载模型和数据
    print("\n加载模型...")
    model, _ = get_model_and_task_type(dataset_name, checkpoint_path=checkpoint_path)
    model = model.to(device)
    print(f"✓ BERT-Large 模型已加载")

    print("\n加载数据...")
    data_loader = get_data_loader(dataset_name, batch_size=batch_size)
    print(f"✓ {dataset_name.upper()} 数据已加载")

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
    print(f"\n计算基线损失...")
    baseline_loss = evaluate_loss(model, eval_batches, device, task_type=task_type)
    print(f"✓ 基线损失: {baseline_loss:.4f}")

    # 累积梯度并计算重要性
    print("\n累积梯度并计算重要性得分...")
    scores = collect_gradients_and_compute_importance(
        model, data_loader, device, num_steps=num_steps, task_type=task_type
    )
    print(f"✓ 已计算重要性得分")

    # 强制垃圾回收
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # 定义实验组
    sparsities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                   0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    # 运行实验
    print("\n" + "=" * 80)
    print("开始剪枝实验")
    print("=" * 80)

    results = []
    all_param_names = list(scores.keys())

    # 定期保存中间结果
    def save_intermediate(results_list, output_path):
        if len(results_list) > 0:
            df = pd.DataFrame(results_list)
            df.to_csv(output_path, index=False)

    intermediate_csv = Path(output_dir) / 'intermediate_results.csv'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 测试Embedding层
    print(f"\n{'='*80}")
    print("测试 Embedding Layer")
    print(f"{'='*80}")

    layer_params = select_layer_params('embedding', -1, all_param_names)
    print(f"  参数数量: {len(layer_params)}")

    # 第一次使用详细调试
    for idx, sparsity in enumerate(sparsities):
        debug_mode = (idx == 0)  # 只在第一次稀疏度时启用详细调试

        result = prune_and_evaluate(
            model, layer_params, scores, sparsity, eval_batches, device, task_type, debug=debug_mode
        )

        result['layer'] = 'embedding'
        result['layer_name'] = 'Embedding Layer'
        result['block_id'] = -1
        result['sparsity'] = sparsity
        result['baseline_loss'] = baseline_loss
        result['loss_increase'] = result['loss'] - baseline_loss

        results.append(result)

        print(f"  稀疏度 {sparsity*100:5.1f}%: "
              f"loss={result['loss']:.4f}, "
              f"Δloss={result['loss_increase']:+.4f}, "
              f"实际剪枝={result.get('actual_sparsity', 0)*100:.1f}%")

    save_intermediate(results, intermediate_csv)

    # 测试所有24个blocks的细粒度层次
    layer_configs = [
        ('attn_q', 'Attention Query'),
        ('attn_k', 'Attention Key'),
        ('attn_v', 'Attention Value'),
        ('attn_out', 'Attention Output'),
        ('ffn_fc', 'FFN FC'),
        ('ffn_proj', 'FFN Proj'),
        ('ln_attn', 'LayerNorm (Attn)'),
        ('ln_ffn', 'LayerNorm (FFN)'),
    ]

    for block_id in range(num_blocks):
        print(f"\n{'='*80}")
        print(f"测试 Block {block_id}")
        print(f"{'='*80}")

        for layer_type, layer_display_name in layer_configs:
            print(f"\n  [Block {block_id}] {layer_display_name}")
            layer_params = select_layer_params(layer_type, block_id, all_param_names)

            if len(layer_params) == 0:
                print(f"    警告: 未找到参数")
                continue

            print(f"    参数: {layer_params}")

            for idx, sparsity in enumerate(sparsities):
                # 只在第一个 block 的第一个层的第一次稀疏度时启用详细调试
                debug_mode = (block_id == 0 and idx == 0 and layer_type == 'attn_q')

                result = prune_and_evaluate(
                    model, layer_params, scores, sparsity, eval_batches, device, task_type, debug=debug_mode
                )

                result['layer'] = layer_type
                result['layer_name'] = f'Block {block_id} - {layer_display_name}'
                result['block_id'] = block_id
                result['sparsity'] = sparsity
                result['baseline_loss'] = baseline_loss
                result['loss_increase'] = result['loss'] - baseline_loss

                results.append(result)

                print(f"    稀疏度 {sparsity*100:5.1f}%: "
                      f"loss={result['loss']:.4f}, "
                      f"Δloss={result['loss_increase']:+.4f}, "
                      f"实际剪枝={result.get('actual_sparsity', 0)*100:.1f}%")

        # 每个block完成后保存并清理内存
        save_intermediate(results, intermediate_csv)
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    # 保存最终结果
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)
    save_results(results, output_dir, baseline_loss, num_blocks=num_blocks, task_type=task_type)

    print("\n" + "=" * 80)
    print("实验完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='BERT-Large 细粒度剪枝分析（只测试 weight 参数）')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['sst2', 'mnli', 'stsb'],
                        help='GLUE 数据集选择')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='梯度累积步数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--num_eval_batches', type=int, default=10,
                        help='评估批次数')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备')
    parser.add_argument('--output_dir', type=str,
                        default='results/bert_fine_grained_pruning_analysis',
                        help='输出目录')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点路径（默认使用 1000 步检查点）')

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = 'cpu'

    # 设置输出目录
    output_dir = f"{args.output_dir}/{args.dataset}"

    run_fine_grained_pruning_experiment(
        dataset_name=args.dataset,
        output_dir=output_dir,
        device=device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        num_eval_batches=args.num_eval_batches,
        checkpoint_path=args.checkpoint
    )


if __name__ == '__main__':
    main()
