"""
Generate Pareto curve: Sparsity vs Quality for SST-2 sweep results.
Uses real experimental data from K=5 sweep.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / 'figs'

# Real SST-2 sweep results (K=5, cosine lr=1e-5, 1000 steps)
oracle_acc = 92.78

data = {
    'Magnitude + Uniform': {
        'sparsity': [0.10, 0.20, 0.30, 0.40],
        'accuracy': [92.43, 92.20, 91.86, 91.97],
        'color': '#1976D2', 'marker': 's', 'ls': '--',
    },
    'First-order + Uniform': {
        'sparsity': [0.10, 0.20, 0.30, 0.40],
        'accuracy': [92.55, 91.97, 92.09, 90.48],
        'color': '#388E3C', 'marker': '^', 'ls': ':',
    },
    'Magnitude + Weibull (Ours)': {
        'sparsity': [0.10, 0.20, 0.30, 0.40],
        'accuracy': [92.20, 92.43, 91.63, 91.97],
        'color': '#D32F2F', 'marker': 'o', 'ls': '-',
    },
}

fig, ax = plt.subplots(figsize=(8, 5))

# Oracle line
ax.axhline(y=oracle_acc, color='gray', linestyle='-', linewidth=1, alpha=0.5,
           label=f'No compression ({oracle_acc}%)')

for name, d in data.items():
    ax.plot(np.array(d['sparsity']) * 100, d['accuracy'],
            color=d['color'], marker=d['marker'], markersize=8,
            linestyle=d['ls'], linewidth=2, label=name)

ax.set_xlabel('Sparsity Ratio (%)')
ax.set_ylabel('SST-2 Accuracy (%)')
ax.set_title('BERT-Large SST-2: Accuracy vs Sparsity (K=5 FT cycles)',
             fontweight='bold')
ax.legend(loc='lower left', framealpha=0.9)
ax.set_xlim(5, 45)
ax.set_ylim(89.5, 93.5)
ax.grid(True, alpha=0.2)

# Annotate first-order collapse
ax.annotate('First-order\ncollapses',
            xy=(40, 90.48), xytext=(33, 89.8),
            fontsize=9, color='#388E3C',
            arrowprops=dict(arrowstyle='->', color='#388E3C'))

plt.tight_layout()
out = OUTPUT_DIR / 'sst2_pareto_curve.pdf'
fig.savefig(str(out), bbox_inches='tight')
fig.savefig(str(out.with_suffix('.png')), bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
