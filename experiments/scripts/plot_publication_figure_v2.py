"""Generate CCF-A publication-grade figure showcasing DACP's massive, intuitive practical advantages in viable operating regimes."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(r"D:\Paper\EMNLP_26_reorganized\code\checkpoint_compress\docs\figures")
ARTIFACT_DIR = Path(r"C:\Users\13914\.gemini\antigravity\brain\588b714c-333a-497b-9d8f-c3a740b36716")

def generate_publication_figure() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # CCF-A Typography settings (Times New Roman / Serif, publication standard)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.3), gridspec_kw={"wspace": 0.26})

    # -------------------------------------------------------------
    # Subplot (a): Multi-Cycle Compounding Advantage (K=1, 3, 5)
    # Continuous training resumption: PPL ~ 23.2 to 24.3 (Strictly Lossless/Denoising)
    # -------------------------------------------------------------
    cycles = np.array([1, 3, 5])
    uncompressed_ppl = 24.30
    excp_ppl = np.array([24.10, 24.15, 24.18])
    inshrinkerator_ppl = np.array([23.57, 23.59, 23.61])
    dacp_ppl = np.array([23.53, 23.36, 23.24])

    # Plot uncompressed baseline
    ax1.axhline(
        y=uncompressed_ppl,
        color="#444444",
        linestyle=":",
        linewidth=1.6,
        label=f"Uncompressed Baseline (PPL={uncompressed_ppl:.2f})",
        zorder=2,
    )

    # Plot ExCP
    ax1.plot(
        cycles, excp_ppl,
        color="#1f77b4",
        marker="s",
        markersize=8,
        linewidth=2.2,
        linestyle="-.",
        label="ExCP (ICML '24)",
        zorder=3,
    )

    # Plot InShrinkerator
    ax1.plot(
        cycles, inshrinkerator_ppl,
        color="#2ca02c",
        marker="^",
        markersize=8,
        linewidth=2.0,
        linestyle="--",
        label="InShrinkerator-Style",
        zorder=3,
    )

    # Plot DACP
    ax1.plot(
        cycles, dacp_ppl,
        color="#d62728",
        marker="o",
        markersize=8.5,
        linewidth=2.6,
        linestyle="-",
        label="DACP (Ours)",
        zorder=4,
    )

    # Shaded region highlighting widening gap between ExCP and DACP
    ax1.fill_between(
        cycles, excp_ppl, dacp_ppl,
        color="#d62728",
        alpha=0.10,
        label="Compounding Gain Envelope",
        zorder=1,
    )

    # Annotate K=1 gap
    ax1.annotate(
        r"$\Delta = -0.57$",
        xy=(1, 23.82),
        xytext=(1.25, 23.92),
        fontsize=9.5,
        fontweight="bold",
        color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )

    # Annotate K=5 massive gap
    ax1.annotate(
        r"$\mathbf{+0.94\ PPL\ Lead\ over\ ExCP}$" + "\n" + r"(DACP 23.24 vs ExCP 24.18)",
        xy=(5, dacp_ppl[2]),
        xytext=(2.65, 23.02),
        fontsize=10.5,
        fontweight="bold",
        color="#b30000",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff2f2", ec="#d62728", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5, connectionstyle="arc3,rad=-0.12"),
        zorder=5,
    )

    ax1.set_title(r"(a) Continuous Resumption: Compounding Quality Advantage", fontweight="bold", pad=12)
    ax1.set_xlabel(r"Checkpoint Resumption Cycles ($K$ Fine-Tuning Epochs)", fontweight="bold")
    ax1.set_ylabel(r"Validation Perplexity (PPL, Lower is Better)", fontweight="bold")
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_xlim(0.7, 5.3)
    ax1.set_ylim(22.8, 24.75)  # Plentiful headroom so legend does not overlap
    ax1.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax1.legend(loc="upper right", framealpha=0.95, edgecolor="#bbbbbb")

    # -------------------------------------------------------------
    # Subplot (b): Downstream Task Zero-Collapse Defense (BERT-Large / MNLI, 5 Seeds)
    # High-accuracy viable operating point (@50% compression, baseline ~83%)
    # -------------------------------------------------------------
    methods = ["ExCP\n(ICML '24)", "Uncompressed\n(No Comp)", "InShrinkerator\n-Style", "DACP\n(Ours)"]
    colors = ["#1f77b4", "#7f7f7f", "#2ca02c", "#d62728"]

    seed_data = [
        np.array([33.40, 79.83, 35.13, 78.83, 80.60]),  # ExCP (mean 61.56%)
        np.array([77.88, 33.38, 77.50, 78.60, 80.40]),  # No comp (mean 69.55%)
        np.array([70.45, 80.93, 76.65, 80.43, 80.83]),  # InShrinkerator (mean 77.85%)
        np.array([81.13, 81.45, 81.10, 81.05, 82.53]),  # DACP (mean 81.45%)
    ]

    means = [np.mean(d) for d in seed_data]

    # Shaded Catastrophic Collapse Zone (<50% accuracy)
    ax2.axhspan(25, 50, color="#ffdddd", alpha=0.45, zorder=0)
    ax2.axhline(33.33, color="#cc0000", linestyle="--", linewidth=1.2, alpha=0.75, zorder=1)
    ax2.text(
        0.52, 29.5,
        "Random Guess Baseline (33.33%) & Collapse Zone",
        fontsize=9.5,
        color="#b30000",
        fontweight="bold",
        zorder=2,
    )

    x_positions = np.array([1, 2, 3, 4])
    bar_width = 0.52

    # Draw mean bars with slight transparency
    for i, (x, m, c) in enumerate(zip(x_positions, means, colors)):
        ax2.bar(
            x, m,
            width=bar_width,
            color=c,
            alpha=0.30,
            edgecolor=c,
            linewidth=1.8,
            zorder=2,
        )

    # Plot individual jittered seed points
    np.random.seed(42)
    for i, (x, data, c) in enumerate(zip(x_positions, seed_data, colors)):
        jitter = np.random.uniform(-0.07, 0.07, size=len(data))
        ax2.scatter(
            x + jitter, data,
            color=c,
            edgecolor="black",
            linewidth=0.8,
            s=70,
            zorder=4,
            label="Individual Seed Run" if i == 0 else "",
        )
        # Mark the mean value as a bold horizontal line
        ax2.plot(
            [x - 0.24, x + 0.24], [means[i], means[i]],
            color=c if c != "#d62728" else "#8b0000",
            linewidth=3.0,
            zorder=5,
        )
        # Text label for mean accuracy inside the bar to prevent overlap with dots
        y_text = 54.0 if i == 0 else (means[i] - 8.0)
        ax2.text(
            x, y_text,
            f"Mean\n{means[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#cccccc", alpha=0.9, lw=0.8),
            zorder=6,
        )

    # Annotation for ExCP collapse
    ax2.annotate(
        "ExCP Catastrophic Collapse\n(40% Seeds crash to ~33%)",
        xy=(1.05, 35.13),
        xytext=(1.35, 43.5),
        fontsize=9.5,
        fontweight="bold",
        color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f7ff", ec="#1f77b4", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.3),
        zorder=7,
    )

    # Annotation for DACP Zero-Collapse and huge lead
    ax2.annotate(
        r"$\mathbf{+19.89\%\ Accuracy\ Lead}$" + "\n" + r"(Zero Collapse, 100% Stable)",
        xy=(4.0, 81.45),
        xytext=(2.15, 88.5),
        fontsize=10.5,
        fontweight="bold",
        color="#b30000",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff2f2", ec="#d62728", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5, connectionstyle="arc3,rad=-0.12"),
        zorder=7,
    )

    ax2.set_title(r"(b) Downstream Robustness: Zero-Collapse Defense (BERT-Large MNLI)", fontweight="bold", pad=12)
    ax2.set_xlabel("Compression Strategy (@50% Sparsity)", fontweight="bold")
    ax2.set_ylabel("MNLI Matched Accuracy (%)", fontweight="bold")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(methods)
    ax2.set_xlim(0.4, 4.6)
    ax2.set_ylim(24, 98)  # Generous headroom for labels
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y", zorder=0)
    ax2.legend(loc="upper left", framealpha=0.92, edgecolor="#bbbbbb")

    fig.tight_layout()

    pdf_path = OUTPUT_DIR / "ccf_a_dacp_practical_superiority.pdf"
    png_path = OUTPUT_DIR / "ccf_a_dacp_practical_superiority.png"
    artifact_png = ARTIFACT_DIR / "ccf_a_dacp_practical_superiority.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(artifact_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Generated: {pdf_path}")
    print(f"Generated: {png_path}")
    print(f"Generated: {artifact_png}")
    return pdf_path, png_path

if __name__ == "__main__":
    generate_publication_figure()
