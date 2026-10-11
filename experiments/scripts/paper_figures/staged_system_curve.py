"""Plot encoded-checkpoint quality from existing H200 records; no new runs."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = []
    for seed in (42, 43, 44):
        inputs += [args.results / f"probe_pythia410m_campaign/system_bytes/excp_dacp_s{seed}.json",
                   args.results / f"probe_pythia410m_campaign/system_bytes_iter/iter_R1_alphas_s{seed}.json",
                   args.results / f"probe_pythia410m_campaign/system_bytes_iter/iter_R1_beyond_s{seed}.json",
                   args.results / f"tolerance/system_bytes/tol_R1_alphas_s{seed}.json",
                   args.results / f"tolerance/system_bytes/tol_R1_fixed_s{seed}.json"]
    methods = {"excp_pipeline": ("ExCP pipeline", "#222222"), "dacp_excpq": ("DACP", "#d97706"),
               "iterative_ucb_excpq": ("Re-linearized + uncertainty", "#0f766e")}
    points, sources, reference = {}, [], []
    for path in inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        sources.append({"path": str(path.relative_to(args.results)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        reference.append(data["reference"]["checkpoint"]["perplexity"])
        for row in data["records"]:
            method = row["method"]
            if method not in methods:
                continue
            op = row["operating_point"]
            key = (method, row["seed"], op["alpha"], op.get("eligible_sparsity"))
            points[key] = (row["total_bytes"]["zlib6"] / 1e6, row["restored"]["perplexity"])
    grouped = {}
    for (method, seed, alpha, sparsity), value in points.items():
        grouped.setdefault((method, alpha, sparsity), []).append(value)
    curves = {}
    for (method, alpha, sparsity), values in grouped.items():
        assert len(values) == 3, (method, alpha, sparsity)
        curves.setdefault(method, []).append({"mb": statistics.mean(v[0] for v in values),
                                             "ppl": statistics.mean(v[1] for v in values),
                                             "std": statistics.stdev(v[1] for v in values)})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    baseline = statistics.mean(reference)
    for axis in axes:
        for method, rows in curves.items():
            rows.sort(key=lambda row: row["mb"])
            label, color = methods[method]
            axis.errorbar([r["mb"] for r in rows], [r["ppl"] for r in rows],
                          yerr=[r["std"] for r in rows], marker="o", markersize=3, linewidth=1.2,
                          label=label, color=color)
        axis.axhline(baseline, color="#64748b", linestyle="--", linewidth=0.9, label="Uncompressed")
        axis.axhline(baseline * 1.01, color="#94a3b8", linestyle=":", linewidth=0.9, label="1% PPL tolerance")
        axis.set_xlabel("Encoded weights + Adam moments (MB)", fontsize=8)
        axis.set_ylabel("Restored perplexity", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.15)
    axes[0].set_xscale("log")
    axes[0].set_xticks([40, 80, 160, 240])
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].set_title("Complete encoding trade-off", fontsize=9)
    axes[1].set_xlim(100, 190)
    axes[1].set_ylim(5.45, 5.9)
    axes[1].set_title("Quality-constrained region", fontsize=9)
    axes[0].legend(fontsize=6, frameon=False)
    figure.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(args.out_dir / f"staged_system_curve.{suffix}", dpi=180)
    (args.out_dir / "staged_system_curve.provenance.json").write_text(
        json.dumps({"sources": sources, "curves": curves, "uncompressed_ppl": baseline}, indent=2), encoding="utf-8")
    print(json.dumps({"source_files": len(sources), "methods": list(curves)}))


if __name__ == "__main__":
    main()
