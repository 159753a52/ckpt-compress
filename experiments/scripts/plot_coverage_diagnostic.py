"""Plot exact coverage curves from coverage_diagnostic.json (no other inputs)."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = Path(args.input)
    r = json.loads(src.read_text())
    states = [("base", "base weights"), ("step100", "step 100"), ("step200", "step 200")]
    sets = [("wikitext_scoring", "WikiText-2 (scoring)"), ("wikitext_val", "WikiText-2 (val)"), ("alpaca", "Alpaca")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
    colors = {"base": "tab:gray", "step100": "tab:blue", "step200": "tab:red"}
    for ax, (key, title) in zip(axes, sets):
        for st, label in states:
            x = r["states"][st][key]
            ms = [int(m) for m in x["expected_mean_coverage_with_replacement"]]
            ax.plot(ms, list(x["expected_mean_coverage_with_replacement"].values()), color=colors[st], marker="o", ms=3,
                    label=f"{label}, sampled (H={x['mean_entropy_nats']:.2f})")
            ax.plot(ms, list(x["mean_coverage_topk"].values()), color=colors[st], ls="--", lw=1, label=f"{label}, top-m")
        for tau in (0.9, 0.95):
            ax.axhline(tau, color="k", lw=0.5, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("rounds m")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("mean covered mass")
    axes[0].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=160)
    prov = {"input": str(src), "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
