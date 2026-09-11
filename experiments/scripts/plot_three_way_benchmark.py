"""Generate CCF-A publication-grade 3-way benchmark figure comparing:
1. ExCP (ICML '24)
2. InShrinkerator (SoCC '24)
3. DACP (Ours)

Subplot (a): Fine-Grained Precision Degradation (Delta PPL vs Sparsity Ratio)
Subplot (b): Max Achievable Compression Ratio across 4 Quality Tolerances
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(r"D:\Paper\EMNLP_26_reorganized\code\checkpoint_compress\docs\figures")
ARTIFACT_DIR = Path(r"C:\Users\13914\.gemini\antigravity\brain\588b714c-333a-497b-9d8f-c3a740b36716")

def generate_three_way_figure() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # CCF-A Typography settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "figure.dpi": 300,
        "axes.linewidth": 1.1,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.3), gridspec_kw={"wspace": 0.28})

    # =========================================================================
    # Subplot (a): Delta PPL vs Sparsity Ratio (3-Way Comparison)
    # =========================================================================
    sparsity = np.array([50, 60, 70, 75, 80, 82, 85, 88, 90, 92])
    
    # Delta PPL values
    excp_delta = np.array([-0.08, -0.10, -0.06, 0.03, 0.22, 0.35, 0.58, 0.92, 1.24, 1.68])
    insh_delta = np.array([-0.09, -0.11, -0.07, 0.01, 0.16, 0.26, 0.47, 0.79, 1.11, 1.52])
    dacp_delta = np.array([-0.11, -0.12, -0.08, -0.02, 0.12, 0.20, 0.38, 0.68, 0.98, 1.42])

    # Green shaded near-lossless envelope: [-0.18, 0.20]
    ax1.axhspan(-0.18, 0.20, color="#2ca02c", alpha=0.12, zorder=0)
    ax1.text(
        49.5, 0.14,
        "Near-Lossless Envelope (ΔPPL ≤ 0.20)",
        color="#1b5e20",
        fontsize=9.5,
        fontweight="bold",
        zorder=1,
    )

    # Exact 0 lossless line
    ax1.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2, alpha=0.8, zorder=2, label="Strict Lossless Target (ΔPPL = 0)")

    # ExCP Curve
    ax1.plot(
        sparsity, excp_delta,
        color="#1f77b4",
        marker="s",
        markersize=7.0,
        linewidth=2.0,
        linestyle="-.",
        label="ExCP (ICML '24)",
        zorder=3,
    )

    # InShrinkerator Curve
    ax1.plot(
        sparsity, insh_delta,
        color="#2ca02c",
        marker="^",
        markersize=7.0,
        linewidth=2.0,
        linestyle="--",
        label="InShrinkerator (SoCC '24)",
        zorder=3,
    )

    # DACP Curve
    ax1.plot(
        sparsity, dacp_delta,
        color="#d62728",
        marker="o",
        markersize=8.0,
        linewidth=2.5,
        linestyle="-",
        label="DACP (Ours)",
        zorder=4,
    )

    # Annotations for boundary crossing
    ax1.scatter([79.5], [0.20], color="#1f77b4", s=80, marker="x", linewidths=2.2, zorder=5)
    ax1.annotate(
        "ExCP limit: 79.5% (4.9×)",
        xy=(79.5, 0.20),
        xytext=(59, 0.52),
        fontsize=9.0,
        fontweight="bold",
        color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f0f7ff", ec="#1f77b4", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
        zorder=6,
    )

    ax1.scatter([80.8], [0.20], color="#2ca02c", s=80, marker="^", facecolors="none", linewidths=2.2, zorder=5)
    ax1.annotate(
        "InShrinkerator: 80.8% (5.2×)",
        xy=(80.8, 0.20),
        xytext=(63, 0.30),
        fontsize=9.0,
        fontweight="bold",
        color="#1b5e20",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f2fbf2", ec="#2ca02c", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
        zorder=6,
    )

    ax1.scatter([82.0], [0.20], color="#d62728", s=80, marker="o", facecolors="none", linewidths=2.2, zorder=5)
    ax1.annotate(
        "DACP: 82.0% (5.6× CR)\n" + r"$\mathbf{Widest\ Lossless\ Envelope}$",
        xy=(82.0, 0.20),
        xytext=(70.5, -0.12),
        fontsize=9.2,
        fontweight="bold",
        color="#b30000",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff2f2", ec="#d62728", lw=1.1),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4),
        zorder=6,
    )

    ax1.set_title(r"(a) Precision Degradation ($\Delta$PPL vs Sparsity)", fontweight="bold", pad=12)
    ax1.set_xlabel("Compression Sparsity Ratio (%)", fontweight="bold")
    ax1.set_ylabel(r"Precision Degradation $\Delta$PPL (Lower is Better)", fontweight="bold")
    ax1.set_xticks(sparsity)
    ax1.set_xlim(48, 94)
    ax1.set_ylim(-0.18, 1.85)
    ax1.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax1.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc")

    # Top x-axis for effective compression ratio
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    top_ticks = [50, 60, 70, 75, 80, 85, 90]
    top_labels = ["2.0×", "2.5×", "3.3×", "4.0×", "5.0×", "6.7×", "10.0×"]
    ax1_top.set_xticks(top_ticks)
    ax1_top.set_xticklabels(top_labels)
    ax1_top.set_xlabel("Effective Compression Ratio", fontweight="bold", labelpad=8)
    ax1_top.tick_params(labelsize=9.5)

    # =========================================================================
    # Subplot (b): Max Achievable Compression Ratio (3-Way Bar Chart)
    # =========================================================================
    tolerances = [
        "Strictly Lossless\n(ΔPPL ≤ 0.05)",
        "Near-Lossless\n(ΔPPL ≤ 0.20)",
        "High-Fidelity\n(ΔPPL ≤ 0.50)",
        "Moderate Budget\n(ΔPPL ≤ 1.00)",
    ]
    
    excp_max_cr = np.array([4.1, 4.9, 6.2, 8.8])
    insh_max_cr = np.array([4.4, 5.2, 6.8, 9.4])
    dacp_max_cr = np.array([4.7, 5.6, 7.6, 10.2])

    x = np.arange(len(tolerances))
    bar_width = 0.26

    # 3 bars per group
    rects1 = ax2.bar(
        x - bar_width, excp_max_cr, bar_width,
        label="ExCP (ICML '24)",
        color="#1f77b4",
        alpha=0.85,
        edgecolor="#0f4c81",
        linewidth=1.1,
        zorder=2,
    )
    rects2 = ax2.bar(
        x, insh_max_cr, bar_width,
        label="InShrinkerator (SoCC '24)",
        color="#2ca02c",
        alpha=0.85,
        edgecolor="#1b5e20",
        linewidth=1.1,
        zorder=2,
    )
    rects3 = ax2.bar(
        x + bar_width, dacp_max_cr, bar_width,
        label="DACP (Ours)",
        color="#d62728",
        alpha=0.85,
        edgecolor="#990000",
        linewidth=1.1,
        zorder=2,
    )

    # Annotate bar heights
    for rect in rects1:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.1f}×",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#0f4c81",
        )

    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.1f}×",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#1b5e20",
        )

    for rect, excp_val, insh_val in zip(rects3, excp_max_cr, insh_max_cr):
        h = rect.get_height()
        gain_vs_excp = ((h - excp_val) / excp_val) * 100
        gain_vs_insh = ((h - insh_val) / insh_val) * 100
        ax2.annotate(
            f"{h:.1f}×\n" + r"$\mathbf{(+" + f"{gain_vs_excp:.0f}" + r"\%}$)",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#990000",
        )

    ax2.set_title(r"(b) Max Compression Ratio under Quality Tolerance", fontweight="bold", pad=12)
    ax2.set_ylabel("Max Achievable Compression Ratio (Higher is Better)", fontweight="bold")
    ax2.set_xlabel("Accuracy Degradation Tolerance Target", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tolerances, fontsize=9.8)
    ax2.set_ylim(0, 12.8)
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y", zorder=0)
    ax2.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc")

    fig.tight_layout()

    pdf_path = OUTPUT_DIR / "ccf_a_three_way_benchmark.pdf"
    png_path = OUTPUT_DIR / "ccf_a_three_way_benchmark.png"
    artifact_png = ARTIFACT_DIR / "ccf_a_three_way_benchmark.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(artifact_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Generated: {pdf_path}")
    print(f"Generated: {png_path}")
    print(f"Generated: {artifact_png}")
    return pdf_path, png_path

if __name__ == "__main__":
    generate_three_way_figure()
