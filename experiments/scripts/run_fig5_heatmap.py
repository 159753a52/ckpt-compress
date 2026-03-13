"""
Fig 5: Sub-layer 剪枝率热力图

展示 Gamma-adaptive 分配给每个 (block, param_group) 的剪枝率。
X 轴: 参数组类型 (QKV, AProj, FC, MProj, LN1, LN2)
Y 轴: Transformer block 编号
颜色: 剪枝率（绿=低/敏感, 红=高/不敏感）

支持多种重要性方法的对比热力图。

运行示例:
    # GPT-2 单方法热力图
    python experiments/scripts/run_fig5_heatmap.py \
        --model gpt2-small --dataset wikitext103 \
        --global_prune_ratio 0.3 \
        --num_steps 100 --device cuda

    # GPT-2 多方法对比
    python experiments/scripts/run_fig5_heatmap.py \
        --model gpt2-small --dataset wikitext103 \
        --global_prune_ratio 0.3 \
        --methods second-order-hvp+gamma-adaptive,first-order+gamma-adaptive \
        --device cuda

    # BERT-Large
    python experiments/scripts/run_fig5_heatmap.py \
        --model bert-large --dataset sst2 \
        --checkpoint checkpoints/bert_large_sst2_1000steps/final.pt \
        --global_prune_ratio 0.3 --device cuda

    # 多个全局剪枝率对比
    python experiments/scripts/run_fig5_heatmap.py \
        --model gpt2-small --dataset wikitext103 \
        --global_prune_ratios 0.3,0.5,0.7 \
        --device cuda
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"

import sys
from pathlib import Path
import argparse
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lib.models import load_model
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.results import save_results
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from src.ckpt_compress.pruning.allocation import get_allocation_strategy


# ============================================================
# 参数名解析：从参数名提取 (block_id, param_group)
# ============================================================

def parse_gpt2_param(name):
    """解析 GPT-2 参数名 → (block_id, param_group)
    
    GPT-2 参数名格式:
        transformer.h.{block}.attn.c_attn.weight → (block, attn_qkv)
        transformer.h.{block}.attn.c_proj.weight → (block, attn_proj)
        transformer.h.{block}.mlp.c_fc.weight    → (block, mlp_fc)
        transformer.h.{block}.mlp.c_proj.weight  → (block, mlp_proj)
        transformer.h.{block}.ln_1.weight        → (block, ln_1)
        transformer.h.{block}.ln_2.weight        → (block, ln_2)
    """
    parts = name.split('.')
    if 'transformer' not in parts or 'h' not in parts:
        return None, None
    
    try:
        h_idx = parts.index('h')
        block_id = int(parts[h_idx + 1])
    except (ValueError, IndexError):
        return None, None
    
    if 'attn' in parts and 'c_attn' in parts:
        return block_id, 'attn_qkv'
    elif 'attn' in parts and 'c_proj' in parts:
        return block_id, 'attn_proj'
    elif 'mlp' in parts and 'c_fc' in parts:
        return block_id, 'mlp_fc'
    elif 'mlp' in parts and 'c_proj' in parts:
        return block_id, 'mlp_proj'
    elif 'ln_1' in parts:
        return block_id, 'ln_1'
    elif 'ln_2' in parts:
        return block_id, 'ln_2'
    
    return block_id, None


def parse_bert_param(name):
    """解析 BERT 参数名 → (block_id, param_group)
    
    BERT 参数名格式:
        bert.encoder.layer.{block}.attention.self.query.weight   → (block, attn_q)
        bert.encoder.layer.{block}.attention.self.key.weight     → (block, attn_k)
        bert.encoder.layer.{block}.attention.self.value.weight   → (block, attn_v)
        bert.encoder.layer.{block}.attention.output.dense.weight → (block, attn_out)
        bert.encoder.layer.{block}.intermediate.dense.weight     → (block, ffn_fc)
        bert.encoder.layer.{block}.output.dense.weight           → (block, ffn_proj)
        bert.encoder.layer.{block}.attention.output.LayerNorm.*  → (block, ln_attn)
        bert.encoder.layer.{block}.output.LayerNorm.*            → (block, ln_ffn)
    """
    if 'encoder.layer' not in name:
        return None, None
    
    parts = name.split('.')
    try:
        layer_idx = parts.index('layer')
        block_id = int(parts[layer_idx + 1])
    except (ValueError, IndexError):
        return None, None
    
    if 'attention' in parts and 'self' in parts:
        if 'query' in parts:
            return block_id, 'attn_q'
        elif 'key' in parts:
            return block_id, 'attn_k'
        elif 'value' in parts:
            return block_id, 'attn_v'
    elif 'attention' in parts and 'output' in parts and 'dense' in parts:
        if 'LayerNorm' in name:
            return block_id, 'ln_attn'
        return block_id, 'attn_out'
    elif 'intermediate' in parts and 'dense' in parts:
        return block_id, 'ffn_fc'
    elif 'output' in parts and 'dense' in parts:
        if 'LayerNorm' in name:
            return block_id, 'ln_ffn'
        return block_id, 'ffn_proj'
    elif 'LayerNorm' in name:
        if 'attention' in name:
            return block_id, 'ln_attn'
        else:
            return block_id, 'ln_ffn'
    
    return block_id, None


def parse_resnet_param(name):
    """解析 ResNet 参数名 → (block_id, param_group)"""
    parts = name.split('.')
    
    # layer1.0.conv1.weight → (0, conv1)
    for i, p in enumerate(parts):
        if p.startswith('layer') and len(p) == 6:
            layer_num = int(p[-1])
            block_in_layer = int(parts[i + 1]) if i + 1 < len(parts) else 0
            block_id = (layer_num - 1) * 2 + block_in_layer  # 简化映射
            
            if 'conv1' in parts:
                return block_id, 'conv1'
            elif 'conv2' in parts:
                return block_id, 'conv2'
            elif 'bn1' in parts:
                return block_id, 'bn1'
            elif 'bn2' in parts:
                return block_id, 'bn2'
            elif 'downsample' in parts:
                return block_id, 'downsample'
            return block_id, None
    
    return None, None


def get_param_parser(model_name):
    """根据模型名返回参数名解析器。"""
    if 'gpt2' in model_name:
        return parse_gpt2_param
    elif 'bert' in model_name:
        return parse_bert_param
    elif 'resnet' in model_name:
        return parse_resnet_param
    else:
        return parse_gpt2_param  # 默认


def get_column_order(model_name):
    """根据模型返回参数组的显示顺序和标签。"""
    if 'gpt2' in model_name:
        return (
            ['attn_qkv', 'attn_proj', 'mlp_fc', 'mlp_proj', 'ln_1', 'ln_2'],
            ['Attn QKV', 'Attn Proj', 'MLP FC', 'MLP Proj', 'LN 1', 'LN 2'],
        )
    elif 'bert' in model_name:
        return (
            ['attn_q', 'attn_k', 'attn_v', 'attn_out', 'ffn_fc', 'ffn_proj', 'ln_attn', 'ln_ffn'],
            ['Attn Q', 'Attn K', 'Attn V', 'Attn Out', 'FFN FC', 'FFN Proj', 'LN Attn', 'LN FFN'],
        )
    elif 'resnet' in model_name:
        return (
            ['conv1', 'conv2', 'bn1', 'bn2', 'downsample'],
            ['Conv1', 'Conv2', 'BN1', 'BN2', 'Downsample'],
        )
    else:
        return None, None


# ============================================================
# 聚合和绘图
# ============================================================

def compute_sublayer_rates(layer_ratios, model_name):
    """将逐参数的剪枝率聚合到 (block_id, param_group) 级别。
    
    返回: DataFrame [block_id, param_group, prune_rate]
    """
    parser = get_param_parser(model_name)
    rows = []
    
    for name, ratio in layer_ratios.items():
        block_id, param_group = parser(name)
        if block_id is not None and param_group is not None:
            rows.append({
                'block_id': block_id,
                'param_group': param_group,
                'prune_rate': ratio,
            })
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    # 同一 (block, group) 可能有多个参数（weight + bias），取均值
    agg = df.groupby(['block_id', 'param_group'])['prune_rate'].mean().reset_index()
    return agg


def plot_single_heatmap(agg_df, model_name, global_ratio, method_name, output_dir):
    """绘制单个方法的热力图。"""
    col_order, col_labels = get_column_order(model_name)
    
    pivot = agg_df.pivot(index='block_id', columns='param_group', values='prune_rate')
    
    # 按逻辑顺序排列列
    if col_order:
        pivot = pivot[[c for c in col_order if c in pivot.columns]]
    
    # 转为百分比
    pivot_pct = pivot * 100
    
    n_blocks = len(pivot_pct)
    fig_height = max(6, n_blocks * 0.4 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Pruning Rate (%)'},
        vmin=0,
        vmax=max(pivot_pct.max().max() * 1.1, global_ratio * 100 * 1.5),
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
    )
    
    ax.set_xlabel('Parameter Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Block', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Layer-wise Pruning Rates: {method_name}\n'
        f'(Global Target: {global_ratio*100:.0f}%)',
        fontsize=13, fontweight='bold',
    )
    
    if col_labels:
        visible_labels = [col_labels[col_order.index(c)] for c in pivot.columns if c in col_order]
        ax.set_xticklabels(visible_labels, rotation=30, ha='right')
    
    plt.tight_layout()
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    safe_name = method_name.replace('+', '_')
    ratio_str = f"{int(global_ratio*100)}pct"
    
    png_path = out_dir / f'fig5_heatmap_{safe_name}_{ratio_str}.png'
    pdf_path = out_dir / f'fig5_heatmap_{safe_name}_{ratio_str}.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"  ✓ 热力图: {png_path}")
    plt.close()
    
    # 保存 CSV
    csv_path = out_dir / f'fig5_rates_{safe_name}_{ratio_str}.csv'
    pivot_pct.to_csv(csv_path)
    print(f"  ✓ CSV: {csv_path}")


def plot_comparison_heatmap(all_agg, model_name, global_ratio, output_dir):
    """多方法对比热力图（并排子图）。"""
    n_methods = len(all_agg)
    if n_methods == 0:
        return
    
    col_order, col_labels = get_column_order(model_name)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 8), sharey=True)
    if n_methods == 1:
        axes = [axes]
    
    vmax = 0
    pivots = {}
    for method_name, agg_df in all_agg.items():
        pivot = agg_df.pivot(index='block_id', columns='param_group', values='prune_rate')
        if col_order:
            pivot = pivot[[c for c in col_order if c in pivot.columns]]
        pivots[method_name] = pivot * 100
        vmax = max(vmax, pivot.max().max() * 100)
    
    vmax = max(vmax * 1.1, global_ratio * 100 * 1.5)
    
    for idx, (method_name, pivot_pct) in enumerate(pivots.items()):
        ax = axes[idx]
        sns.heatmap(
            pivot_pct,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            vmin=0, vmax=vmax,
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            cbar=(idx == n_methods - 1),
            cbar_kws={'label': 'Pruning Rate (%)'} if idx == n_methods - 1 else {},
        )
        ax.set_title(method_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Parameter Group', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Block', fontsize=10)
        
        if col_labels:
            visible = [col_labels[col_order.index(c)] for c in pivot_pct.columns if c in col_order]
            ax.set_xticklabels(visible, rotation=30, ha='right', fontsize=9)
    
    fig.suptitle(
        f'Layer-wise Pruning Rate Comparison (Global: {global_ratio*100:.0f}%)',
        fontsize=14, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratio_str = f"{int(global_ratio*100)}pct"
    plt.savefig(out_dir / f'fig5_comparison_{ratio_str}.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / f'fig5_comparison_{ratio_str}.pdf', bbox_inches='tight')
    print(f"  ✓ 对比热力图: {out_dir / f'fig5_comparison_{ratio_str}.png'}")
    plt.close()


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Fig 5: Sub-layer 剪枝率热力图')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--global_prune_ratio', type=float, default=0.3,
                        help='单个全局剪枝率')
    parser.add_argument('--global_prune_ratios', type=str, default=None,
                        help='多个全局剪枝率（逗号分隔），覆盖 --global_prune_ratio')
    parser.add_argument('--methods', type=str,
                        default='second-order-hvp+gamma-adaptive',
                        help='方法列表（逗号分隔），如 second-order-hvp+gamma-adaptive,first-order+gamma-adaptive')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--hvp_batches', type=int, default=8)
    parser.add_argument('--hvp_mode', type=str, default='full', choices=['full', 'block'])
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/paper_results/fig5')
    args = parser.parse_args()
    
    raw_methods = [m.strip() for m in args.methods.split(',')]
    methods = []
    for m in raw_methods:
        parts = m.split('+')
        if len(parts) != 2:
            raise ValueError(f"Invalid method format: {m}. Expected <importance>+<allocation>.")
        imp, alloc = parts[0], parts[1]
        # 兼容旧写法：second-order -> second-order-hvp
        if imp == 'second-order':
            imp = 'second-order-hvp'
        methods.append(f"{imp}+{alloc}")
    
    if args.global_prune_ratios:
        ratios = [float(r) for r in args.global_prune_ratios.split(',')]
    else:
        ratios = [args.global_prune_ratio]
    
    print("=" * 70)
    print("Fig 5: Sub-layer 剪枝率热力图")
    print(f"模型: {args.model} | 数据集: {args.dataset}")
    print(f"方法: {methods}")
    print(f"全局剪枝率: {ratios}")
    print("=" * 70)
    
    # 加载模型和数据
    print("\n[1/3] 加载模型和数据...")
    model, _ = load_model(
        args.model,
        pretrained=True,
        checkpoint_path=args.checkpoint,
        device='cpu',
        dataset_name=args.dataset,
    )
    train_loader, _, task_type = get_data_loaders(
        args.model, args.dataset, args.batch_size, args.seq_length)
    
    cached_train = cache_batches(train_loader, args.num_steps, task_type)
    
    # 预计算得分
    print("[2/3] 计算重要性得分...")
    importance_methods = sorted({m.split('+')[0] for m in methods})
    model_for_scoring = copy.deepcopy(model).to(args.device)
    score_cache = compute_scores_by_method(
        model_for_scoring,
        cached_train,
        task_type,
        methods=importance_methods,
        alpha=args.alpha,
        hvp_batches=args.hvp_batches,
        hvp_mode=args.hvp_mode,
        chunk_size=args.chunk_size,
    )
    del model_for_scoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 生成热力图
    print("[3/3] 生成热力图...")
    all_results = []
    
    for ratio in ratios:
        print(f"\n--- 全局剪枝率: {ratio:.0%} ---")
        all_agg = {}
        
        for method_str in methods:
            parts = method_str.split('+')
            imp, alloc = parts[0], parts[1]
            
            allocator = get_allocation_strategy(alloc)
            scores = score_cache[imp]
            layer_ratios = allocator.allocate(scores, ratio)
            
            # 聚合到 sub-layer 级别
            agg_df = compute_sublayer_rates(layer_ratios, args.model)
            all_agg[method_str] = agg_df
            
            # 单方法热力图
            plot_single_heatmap(agg_df, args.model, ratio, method_str, args.output_dir)
            
            # 收集结果
            for _, row in agg_df.iterrows():
                all_results.append({
                    'method': method_str,
                    'global_ratio': ratio,
                    'block_id': int(row['block_id']),
                    'param_group': row['param_group'],
                    'prune_rate': row['prune_rate'],
                    'prune_rate_pct': row['prune_rate'] * 100,
                })
            
            # 打印摘要
            print(f"  {method_str}:")
            col_order, _ = get_column_order(args.model)
            if col_order:
                for pg_name in col_order:
                    pg_data = agg_df[agg_df['param_group'] == pg_name]['prune_rate']
                    if len(pg_data) > 0:
                        print(f"    {pg_name:<12}: min={pg_data.min()*100:.1f}%, "
                              f"max={pg_data.max()*100:.1f}%, mean={pg_data.mean()*100:.1f}%")
        
        # 多方法对比热力图
        if len(methods) > 1:
            plot_comparison_heatmap(all_agg, args.model, ratio, args.output_dir)
    
    # 保存结果
    save_results(all_results, args.output_dir,
                 f"fig5_{args.model}_{args.dataset}", vars(args))
    
    print("\n实验完成！")


if __name__ == '__main__':
    main()
