"""Generate CCF-A publication-quality figure: Near-Lossless Envelope and Extreme Compression Showdown.

Outputs both PDF (vector) and high-res PNG (300 DPI).
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set publication style fonts
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "figure.titlesize": 14,
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#E0E0E0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.dpi": 300,
})

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\13914\.gemini\antigravity\brain\588b714c-333a-497b-9d8f-c3a740b36716")

# Load real V100 experiment data
sweep_file = ROOT / "experiments/results/v100_lossless_frontier_sweep.json"
e2e_file = ROOT / "experiments/results/v100_dacp_e2e_showdown.json"

with open(sweep_file, "r", encoding="utf-8") as f:
    sweep_data = json.load(f)
with open(e2e_file, "r", encoding="utf-8") as f:
    e2e_data = json.load(f)

records_sweep = sweep_data["records"]
w1_ppl = sweep_data["metadata"]["w1_ppl"]  # 26.82

# Colors
COLOR_DACP = "#D32F2F"    # Deep Academic Crimson Red
COLOR_EXCP = "#1976D2"    # Academic Royal Blue
COLOR_ORACLE = "#424242"  # Charcoal Grey
COLOR_ZONE = "#E8F5E9"    # Soft Pastel Green for Lossless Zone
COLOR_ALERT = "#FFEBEE"   # Soft Pink/Red for Collapse Zone

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)

# =========================================================================
# Subplot (a): Near-Lossless Envelope (Zoomed in on 50% ~ 92% Sparsity)
# =========================================================================
# Filter records up to 92%
sub_recs = [r for r in records_sweep if r["sparsity"] <= 0.92]
sp_a = [r["sparsity"] * 100 for r in sub_recs]
excp_ppl_a = [r["excp_ppl"] for r in sub_recs]
dacp_ppl_a = [r["dacp_ppl"] for r in sub_recs]

# Shaded near-lossless zone (Δ PPL <= 0.20)
lossless_ceiling = w1_ppl + 0.20
ax1.axhspan(w1_ppl - 0.20, lossless_ceiling, color=COLOR_ZONE, alpha=0.85, zorder=1,
            label=r"Near-Lossless Envelope ($\Delta\mathrm{PPL} \leq 0.20$)")
ax1.axhline(w1_ppl, color=COLOR_ORACLE, linestyle="--", linewidth=1.2, zorder=2,
            label=f"Uncompressed $W_1$ Target (PPL={w1_ppl:.2f})")

# Plot curves
ax1.plot(sp_a, excp_ppl_a, color=COLOR_EXCP, marker="s", markersize=6.5, markeredgewidth=1.2,
         linewidth=1.8, linestyle="-.", label="ExCP (ICML '24)", zorder=4)
ax1.plot(sp_a, dacp_ppl_a, color=COLOR_DACP, marker="o", markersize=7, markeredgewidth=1.4,
         linewidth=2.2, linestyle="-", label=r"$\bf{DACP\ (Ours)}$", zorder=5)

# Annotation for Near-lossless boundary extension
ax1.annotate(
    "DACP Widens Lossless Frontier\nto 80% (5.0× CR) with Δ ≤ 0.20",
    xy=(80.0, 27.02), xytext=(58, 27.50),
    arrowprops=dict(facecolor=COLOR_DACP, edgecolor=COLOR_DACP, arrowstyle="->", lw=1.3),
    fontsize=9.5, fontweight="bold", color="#B71C1C",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF9C4", edgecolor="#FBC02D", alpha=0.9)
)

ax1.set_title("(a) Fine-Grained Near-Lossless Compression Frontier", fontweight="bold", pad=10)
ax1.set_xlabel("Sparsity Ratio (%)", fontweight="bold")
ax1.set_ylabel("Perplexity (PPL, Lower is Better)", fontweight="bold")
ax1.set_xlim(48, 93)
ax1.set_ylim(26.5, 28.7)
ax1.grid(True, zorder=0)
ax1.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

# Secondary X-axis for Compression Ratio
ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim())
top_ticks = [50, 60, 70, 75, 80, 85, 90]
ax1_top.set_xticks(top_ticks)
ax1_top.set_xticklabels([f"{1/(1-t/100):.1f}×" for t in top_ticks], fontsize=9.5, color="#555555")
ax1_top.set_xlabel("Effective Residual Compression Ratio", color="#555555", fontsize=10.5, labelpad=6)


# =========================================================================
# Subplot (b): End-to-End Compression (80% ~ 99.5% with 4-bit Quantization + Gzip)
# =========================================================================
sp_b = [r["sparsity"] * 100 for r in e2e_data]
excp_ppl_b = [r["excp_quant_ppl"] for r in e2e_data]
dacp_ppl_b = [r["dacp_quant_ppl"] for r in e2e_data]
w1_b_ppl = 27.19

# Shaded Collapse zone for ExCP
ax2.axvspan(95.0, 100.0, color=COLOR_ALERT, alpha=0.6, zorder=1,
            label="ExCP Catastrophic Collapse Regime")
ax2.axhline(w1_b_ppl, color=COLOR_ORACLE, linestyle="--", linewidth=1.2, zorder=2,
            label=f"Uncompressed $W_1$ Target (PPL={w1_b_ppl:.2f})")

ax2.plot(sp_b, excp_ppl_b, color=COLOR_EXCP, marker="s", markersize=7, markeredgewidth=1.3,
         linewidth=2.0, linestyle="-.", label="ExCP + 4bit K-means (ICML '24)", zorder=4)
ax2.plot(sp_b, dacp_ppl_b, color=COLOR_DACP, marker="*", markersize=10, markeredgewidth=1.4,
         linewidth=2.4, linestyle="-", label=r"$\bf{DACP + 4bit\ K\text{-}means\ (Ours)}$", zorder=5)

# Annotation for 200x compression win (+3.50 PPL)
ax2.annotate(
    r"$\bf{+3.50\ PPL\ Advantage}$" + "\nat 200× (99.5%) Compression\n(DACP 31.14 vs ExCP 34.64)",
    xy=(99.5, 31.14), xytext=(88.5, 34.2),
    arrowprops=dict(facecolor=COLOR_DACP, edgecolor=COLOR_DACP, arrowstyle="->", lw=1.4),
    fontsize=9.5, color="#B71C1C", linespacing=1.25,
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFEBEE", edgecolor="#E57373", alpha=0.95, lw=1.0)
)

# Annotation for Plateau stability
ax2.annotate(
    "DACP Plateau Defense\n(Hessian + Layer Floor)",
    xy=(98.0, 31.14), xytext=(84.0, 28.6),
    arrowprops=dict(facecolor="#2E7D32", edgecolor="#2E7D32", arrowstyle="->", lw=1.3),
    fontsize=9.0, color="#1B5E20", linespacing=1.2,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F5E9", edgecolor="#81C784", alpha=0.92, lw=1.0)
)

ax2.set_title(r"(b) End-to-End Compression (Pruning + 4-bit Quantization)", fontweight="bold", pad=10)
ax2.set_xlabel("Sparsity Ratio (%)", fontweight="bold")
ax2.set_ylabel("Perplexity (PPL, Lower is Better)", fontweight="bold")
ax2.set_xlim(78, 100.2)
ax2.set_ylim(26.5, 35.8)
ax2.grid(True, zorder=0)
ax2.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

# Secondary X-axis for Compression Ratio
ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())
top_ticks_b = [80, 90, 95, 98, 99, 99.5]
ax2_top.set_xticks(top_ticks_b)
ax2_top.set_xticklabels(["5×", "10×", "20×", "50×", "100×", "200×"], fontsize=9.2, color="#555555")
ax2_top.set_xlabel("End-to-End Physical Compression Ratio", color="#555555", fontsize=10.5, labelpad=6)

# Save figure in repo and artifact dir
png_repo = FIGURE_DIR / "ccf_a_lossless_envelope_comparison.png"
pdf_repo = FIGURE_DIR / "ccf_a_lossless_envelope_comparison.pdf"
png_artifact = ARTIFACT_DIR / "ccf_a_lossless_envelope_comparison.png"

plt.savefig(png_repo, dpi=300, bbox_inches="tight")
plt.savefig(pdf_repo, format="pdf", bbox_inches="tight")
plt.savefig(png_artifact, dpi=300, bbox_inches="tight")
plt.close()

print(f"Publication-grade figures successfully generated:")
print(f"  PNG (Repo):     {png_repo}")
print(f"  PDF (Repo):     {pdf_repo}")
print(f"  PNG (Artifact): {png_artifact}")
