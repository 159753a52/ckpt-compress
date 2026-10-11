"""Plot actual continuation after decoded weights and retained BF16 Adam state."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import style as S


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoded-chain", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--pruned-chain", type=Path, required=True)
    args = ap.parse_args()
    paths = [args.encoded_chain, args.reference, args.pruned_chain]
    records = [json.loads(p.read_text()) for p in paths]
    encoded, reference, pruned = records
    assert all(r["status"] == "completed" and r["seed"] == 42 for r in records)
    assert len({r["protocol"]["checkpoint_sha256"] for r in records}) == 1
    assert len({r["identity"]["evaluation_sha256"] for r in records}) == 1
    assert len({r["identity"]["recovery"]["chunk_sha256"][0] for r in records}) == 1
    traces = [r["cycles"][0]["validation_trace"] for r in records]
    steps = [x["step"] for x in traces[0]]
    assert all([x["step"] for x in trace] == steps for trace in traces) and len(steps) == 11
    base = [x["val"]["loss"] for x in traces[1]]
    gaps = [[100 * math.expm1(x["val"]["loss"] - b) for x, b in zip(trace, base)] for trace in traces]
    summary = {"initial_ppl_change_percent": gaps[0][0], "maximum_ppl_change_percent": max(gaps[0]),
               "final_ppl_change_percent": gaps[0][-1], "measured_validation_points": len(steps)}
    S.setup()
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE_W, 2.2))
    styles = [("DACP archive: BF16 Adam", S.C["vermillion"], "*", "-"),
              ("No compression", S.C["black"], "o", "-"),
              ("DACP pruning: FP32 Adam", "#8C8C8C", "s", "--")]
    for trace, gap, (label, color, marker, ls) in zip(traces, gaps, styles):
        kw = dict(label=label, color=color, marker=marker, markersize=3.5, linewidth=1.0, linestyle=ls)
        axes[0].plot([s - steps[0] for s in steps], [x["val"]["ppl"] for x in trace], **kw)
        axes[1].plot([s - steps[0] for s in steps], gap, **kw)
    axes[1].axhline(1, color="#666666", ls="--", lw=.8)
    axes[0].set(title="(a) Validation perplexity", ylabel="Perplexity")
    axes[1].set(title="(b) Quality change vs. uncompressed", ylabel="Relative perplexity change (%)")
    for ax in axes:
        ax.set_xlabel("Recovery update")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.03))
    fig.tight_layout(rect=(0, .10, 1, 1))
    out = Path(__file__).resolve().parents[5] / "paper/figs/gen"
    S.save(fig, out, "v100_encoded_recovery", inputs=paths, notes={"summary": summary,
           "protocol": "one 100-update continuation, measured every ten updates; original uncompressed reference reused"})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
