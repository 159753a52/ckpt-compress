"""Plot completed dense V100 recovery records; verify pairing before plotting."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from make_figures import OUT, REPO
import matplotlib.pyplot as plt

ORDER = ["no_compression", "excp_style", "inshrinkerator_style", "dacp", "first_order_ucb", "iterative_ucb_disjoint"]
S.METHODS.update({
    "first_order_ucb": S.MethodStyle("Protected ($1\\times16$)", S.C["green"], "d"),
    "iterative_ucb_disjoint": S.MethodStyle("Ours: staged + protected ($4\\times4$)", S.C["vermillion"], "*"),
})


def load(root):
    records, paths = {}, []
    for path in sorted(root.glob("*.json")):
        r = json.loads(path.read_text())
        if "method" in r:
            if r.get("status") != "completed" or r["method"] in records:
                raise ValueError("incomplete or repeated chain")
            records[r["method"]] = r
            paths.append(path)
    if set(records) != set(ORDER):
        raise ValueError("six completed chains are required")
    nc = records["no_compression"]
    config = nc["protocol"]["config"]
    for r in records.values():
        if r["protocol"] != nc["protocol"] or r["identity"] != nc["identity"]:
            raise ValueError("protocol or data pairing differs")
        if len(r["cycles"]) != config["cycles"] or r["seed"] != 42:
            raise ValueError("cycle count or seed differs")
        for c, ref in zip(r["cycles"], nc["cycles"]):
            if [v["step"] for v in c["validation_trace"]] != [v["step"] for v in ref["validation_trace"]]:
                raise ValueError("validation steps differ")
            if [v["step"] for v in c["train_log"]] != [v["step"] for v in ref["train_log"]] or len(c["train_log"]) != config["recovery"]:
                raise ValueError("training steps differ")
    return records, paths


def trajectory(record, nc=None):
    xs, ys = [], []
    previous = record["uncompressed"]["val"]["loss"]
    previous_nc = nc["uncompressed"]["val"]["loss"] if nc else 0
    for c in record["cycles"]:
        reference = nc["cycles"][c["cycle"] - 1] if nc else None
        xs.append(c["step"])
        ys.append(previous - previous_nc)
        for i, value in enumerate(c["validation_trace"]):
            xs.append(value["step"])
            ys.append(value["val"]["loss"] - (reference["validation_trace"][i]["val"]["loss"] if nc else 0))
        previous = c["final"]["val"]["loss"]
        if nc:
            previous_nc = reference["final"]["val"]["loss"]
    return xs, [v * 1e3 for v in ys] if nc else ys


def summarize(records):
    nc = records["no_compression"]
    report = {"passed_pairing_checks": True, "seed": 42, "tolerance_relative_ppl": 0.01, "methods": {}}
    threshold = math.log1p(0.01)
    for method in ORDER:
        if method == "no_compression":
            continue
        cycles = []
        for c, ref in zip(records[method]["cycles"], nc["cycles"]):
            gaps = [v["val"]["loss"] - n["val"]["loss"] for v, n in zip(c["validation_trace"], ref["validation_trace"])]
            offsets = [v["step"] - c["step"] for v in c["validation_trace"]]
            stable = next((offsets[i] for i in range(len(gaps)) if all(v <= threshold for v in gaps[i:])), None)
            cycles.append({"cycle": c["cycle"], "immediate_gap_millinats": gaps[0] * 1e3,
                           "final_gap_millinats": gaps[-1] * 1e3,
                           "first_measured_step_with_all_later_points_within_tolerance": stable,
                           "validation_measurements": len(gaps), "training_updates": len(c["train_log"])})
        report["methods"][method] = cycles
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "results/v100_dense_recovery_20261011")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    records, paths = load(args.root)
    nc = records["no_compression"]
    S.setup()
    fig, axes = plt.subplots(2, 2, figsize=(S.DOUBLE_W, 4.9))
    val, gap, train, aligned = axes.flat
    for method in ORDER:
        r = records[method]
        xs, ys = trajectory(r)
        val.plot(xs, ys, **S.line_kw(method, ms=2.7, lw=1))
        if method == "no_compression":
            continue
        xs, ys = trajectory(r, nc)
        gap.plot(xs, ys, **S.line_kw(method, ms=2.7, lw=1))
        by_cycle = []
        for c, ref in zip(r["cycles"], nc["cycles"]):
            steps = [v["step"] for v in c["train_log"]]
            differences = [(v["loss"] - n["loss"]) * 1e3 for v, n in zip(c["train_log"], ref["train_log"])]
            train.plot(steps, differences, color=S.style_of(method).color, alpha=0.18, lw=0.4)
            train.plot(steps, S.moving_average(differences, 10), **S.line_kw(method, marker=False, lw=1))
            offsets = [v["step"] - c["step"] for v in c["validation_trace"]]
            differences = [(v["val"]["loss"] - n["val"]["loss"]) * 1e3 for v, n in zip(c["validation_trace"], ref["validation_trace"])]
            by_cycle.append(differences)
            aligned.plot(offsets, differences, color=S.style_of(method).color, alpha=0.18, lw=0.5)
        mean = [sum(v) / len(v) for v in zip(*by_cycle)]
        aligned.plot(offsets, mean, **S.line_kw(method, ms=2.7, lw=1))
    titles = ["(a) Validation loss", "(b) Validation gap vs. uncompressed",
              "(c) Training gap: raw and ten-step mean", "(d) Recovery gap: mean of three cycles"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Training step" if ax is not aligned else "Recovery step within cycle")
        if ax is not aligned:
            for c in nc["cycles"]:
                ax.axvline(c["step"], color="0.7", ls=":", lw=0.6, zorder=0)
    val.set_ylabel("Validation loss (nats)")
    gap.set_ylabel(r"Validation $\Delta$ ($10^{-3}$ nats)")
    train.set_ylabel(r"Training $\Delta$ ($10^{-3}$ nats)")
    aligned.set_ylabel(r"Validation $\Delta$ ($10^{-3}$ nats)")
    for ax in (gap, train, aligned):
        ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    for ax in (gap, aligned):
        ax.axhline(math.log1p(0.01) * 1e3, color="0.4", ls="--", lw=0.6, zorder=0)
    handles, labels = val.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=1)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5, frameon=False)
    S.save(fig, args.out, "v100_dense_recovery", paths,
           notes={"validation": "fixed 100-batch evaluation every ten updates; markers measured",
                  "training": "raw paired per-update loss differences and trailing ten-step means; reset per cycle",
                  "aligned": "mean across three consecutive cycles of one seed; faint curves are individual cycles, not confidence bounds",
                  "threshold": "one percent relative perplexity tolerance: log(1.01)",
                  "sparsity": 0.98, "seed": 42})
    report = summarize(records)
    (args.root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
