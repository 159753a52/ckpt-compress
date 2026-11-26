"""生成 Weibull 拟合优势的 Box Plot 对比图（论文级，简约风格）"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'

def load_ks(path):
    with open(path, 'r') as f:
        data = json.load(f)
    dists = ['weibull', 'lognormal', 'exponential', 'gamma']
    return {d: [layer[f'{d}_ks'] for layer in data] for d in dists}

ks_gpt2 = load_ks('gpt2m_fit_results.json')
ks_bert = load_ks('bert_fit_results.json')

dists = ['weibull', 'lognormal', 'exponential', 'gamma']
labels = ['Weibull', 'Log-Normal', 'Exponential', 'Gamma']
# 低饱和度学术配色
colors = ['#4878A8', '#6AAF6A', '#D4955A', '#C06060']

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=False)

for ax, ks_data, title in [
    (axes[0], ks_gpt2, '(a) GPT-2 Medium'),
    (axes[1], ks_bert, '(b) BERT-Large'),
]:
    data_list = [ks_data[d] for d in dists]
    
    bp = ax.boxplot(data_list, positions=range(len(dists)), widths=0.45,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.2),
                    whiskerprops=dict(color='#555', linewidth=0.8),
                    capprops=dict(color='#555', linewidth=0.8))
    
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.55)
        patch.set_edgecolor(colors[i])
        patch.set_linewidth(1.0)
    
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_ylabel('KS-D Statistic' if ax == axes[0] else '', fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

plt.tight_layout(w_pad=1.5)
plt.savefig('fig_distribution_fit_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_distribution_fit_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: fig_distribution_fit_comparison.pdf / .png")
