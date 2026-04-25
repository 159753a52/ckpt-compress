"""
生成 Weibull 分布拟合可视化图。
选 4 个代表性层，画直方图 + 多分布拟合曲线对比。
"""

import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lib.models import load_model
from experiments.lib.data import get_data_loaders, cache_batches
from experiments.lib.importance_compare.scoring import compute_scores_by_method


def plot_weibull_fit():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载 GPT-2 Small
    print("Loading model...")
    model, model_family = load_model('gpt2-small', device)

    # 加载数据
    print("Loading data...")
    train_loader, eval_loader, task_type = get_data_loaders(
        'gpt2-small', 'wikitext2', batch_size=2, seq_length=128,
        data_dir=str(ROOT / 'data')
    )
    cached_train = cache_batches(train_loader, 10, task_type)

    # 计算 damage scores
    print("Computing damage scores (block-wise HVP)...")
    score_cache = compute_scores_by_method(
        model, cached_train, task_type,
        methods=['second-order-hvp'],
        alpha=0.5,
        hvp_batches=2,
        hvp_mode='block',
        model_family=model_family,
    )
    scores = score_cache['second-order-hvp']
    
    # 选 4 个代表性层（不同类型）
    layer_names = list(scores.keys())
    # 选 attn, mlp, ln 各一个 + 一个深层
    selected = []
    for name in layer_names:
        if 'h.0.attn.c_attn' in name and len(selected) < 1:
            selected.append(name)
        elif 'h.0.mlp.c_fc' in name and len(selected) < 2:
            selected.append(name)
        elif 'h.5.attn.c_attn' in name and len(selected) < 3:
            selected.append(name)
        elif 'h.10.mlp.c_proj' in name and len(selected) < 4:
            selected.append(name)
    
    if len(selected) < 4:
        # fallback: 取前 4 层
        selected = layer_names[:4]
    
    print(f"Selected layers: {selected}")
    
    # 画图：2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weibull Distribution Fitting on Damage Scores (GPT-2 Small)', 
                 fontsize=14, fontweight='bold')
    
    for idx, (ax, name) in enumerate(zip(axes.flat, selected)):
        data = scores[name].flatten().float().cpu().numpy()
        data_pos = data[data > 0]
        
        # 子采样
        rng = np.random.RandomState(42)
        if len(data_pos) > 50000:
            data_pos = rng.choice(data_pos, 50000, replace=False)
        
        # 拟合四种分布
        weibull_params = stats.weibull_min.fit(data_pos, floc=0)
        gamma_mean = data_pos.mean()
        gamma_var = data_pos.var()
        gamma_k = gamma_mean ** 2 / max(gamma_var, 1e-30)
        gamma_theta = gamma_var / max(gamma_mean, 1e-30)
        gamma_params = (gamma_k, 0, gamma_theta)
        lognorm_params = stats.lognorm.fit(data_pos, floc=0)
        expon_params = stats.expon.fit(data_pos, floc=0)
        
        # KS 统计量
        weibull_ks = stats.kstest(data_pos, 'weibull_min', args=weibull_params).statistic
        gamma_ks = stats.kstest(data_pos, 'gamma', args=gamma_params).statistic
        lognorm_ks = stats.kstest(data_pos, 'lognorm', args=lognorm_params).statistic
        expon_ks = stats.kstest(data_pos, 'expon', args=expon_params).statistic
        
        # 用 CDF 画图（KS-D 就是经验 CDF 与理论 CDF 的最大垂直距离）
        sorted_data = np.sort(data_pos)
        ecdf_y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 经验 CDF（降采样以加速绘制）
        step = max(1, len(sorted_data) // 2000)
        ax.plot(sorted_data[::step], ecdf_y[::step], 
                color='steelblue', linewidth=2, label='Empirical CDF')
        
        # 理论 CDF 曲线
        x = np.linspace(sorted_data[0], sorted_data[-1], 1000)
        
        ax.plot(x, stats.weibull_min.cdf(x, *weibull_params), 
                color='red', linewidth=2.5, 
                label=f'Weibull (KS-D={weibull_ks:.4f})')
        ax.plot(x, stats.gamma.cdf(x, *gamma_params), 
                color='orange', linewidth=2, linestyle='--',
                label=f'Gamma (KS-D={gamma_ks:.4f})')
        ax.plot(x, stats.lognorm.cdf(x, *lognorm_params), 
                color='green', linewidth=1.5, linestyle=':',
                label=f'LogNormal (KS-D={lognorm_ks:.4f})')
        ax.plot(x, stats.expon.cdf(x, *expon_params), 
                color='purple', linewidth=1.5, linestyle='-.',
                label=f'Exponential (KS-D={expon_ks:.4f})')
        
        # 简化层名
        short_name = name.replace('transformer.', '').replace('.weight', '')
        ax.set_title(short_name, fontsize=11)
        ax.set_xlabel('Damage Score')
        ax.set_ylabel('CDF')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    output_path = ROOT / 'results' / 'paper_results' / 'weibull_fit_visualization.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    plot_weibull_fit()
