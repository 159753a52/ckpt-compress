"""Figures and table rows for the 32B multi-checkpoint trajectory study.

Reads results/probe_trajectory_32b/traj_<dataset>/trajectory/step*_vs_*.json
(written by run_trajectory_32b.py).  Outputs grouped bars of the immediate
validation-loss change per checkpoint at selected sparsities and LaTeX rows.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHODS = [
    ("excp_style", "ExCP-style", "#E69F00"),
    ("inshrinkerator_style", "Inshrinkerator-style", "#56B4E9"),
    ("first_order", "First order", "#009E73"),
    ("hvp_global", "Block HVP", "#D55E00"),
    ("dacp", "DACP", "#0072B2"),
    ("probe_sample_m1", "Probe sampled m=1", "#332288"),
    ("probe_headtail_m2", "Probe head-tail m=2", "#882255"),
]


def load(root: Path):
    runs = []
    for p in sorted(root.glob("traj_*/trajectory/step*_vs_*.json")):
        m = re.match(r"step(\d+)_vs_(\w+)", p.stem)
        r = json.loads(p.read_text())
        runs.append({"path": p, "dataset": p.parts[-3].replace("traj_", ""), "step": int(m.group(1)), "ref": m.group(2), "r": r})
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/probe_trajectory_32b")
    ap.add_argument("--out", default="../../paper/figs/gen")
    ap.add_argument("--sparsities", default="0.95,0.98,0.99")
    args = ap.parse_args()
    runs = load(Path(args.root))
    if not runs:
        print("no trajectory records")
        return
    sps = args.sparsities.split(",")
    out = Path(args.out)
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, len(sps), figsize=(6.9, 2.4), sharey=False)
    labels = [f"{x['dataset']}\n{x['step']} vs {x['ref']}" for x in runs]
    width = 0.8 / len(METHODS)
    for ax, sp in zip(axes, sps):
        for j, (key, name, color) in enumerate(METHODS):
            ys = []
            for x in runs:
                ev = x["r"]["evals"].get(key, {}).get(sp)
                ys.append((ev["val"] - x["r"]["uncompressed"]["val"]) * 1e3 if ev else float("nan"))
            ax.bar([i + (j - len(METHODS) / 2 + 0.5) * width for i in range(len(runs))], ys, width, color=color, label=name)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_title(f"P = {float(sp):.0%}", fontsize=8)
    axes[0].set_ylabel(r"$\Delta$ val loss ($10^{-3}$ nats)")
    axes[-1].legend(fontsize=5.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "d1_trajectory_32b.pdf")
    fig.savefig(out / "d1_trajectory_32b.png", dpi=220)
    rows = []
    for x in runs:
        r = x["r"]
        base = r["uncompressed"]["val"]
        best = {sp: min(r["evals"][k][sp]["val"] for k, _, _ in METHODS if k in r["evals"]) for sp in sps}
        head = f"{x['dataset']} {x['step']}/{x['ref'].replace('step', '')}"
        for key, name, _ in METHODS:
            if key not in r["evals"]:
                continue
            cells = []
            for sp in sps:
                v = r["evals"][key][sp]["val"]
                t = f"{(v - base) * 1e3:+.1f}"
                cells.append(f"$\\mathbf{{{t}}}$" if v == best[sp] else f"${t}$")
            rows.append(f"{head} & {name} & " + " & ".join(cells) + " \\\\")
            head = ""
        rows.append("\\midrule")
    (out / "d1_trajectory_rows.tex").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    prov = {"inputs": {str(x["path"]): hashlib.sha256(x["path"].read_bytes()).hexdigest() for x in runs},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "d1_trajectory_32b.provenance.json").write_text(json.dumps(prov, indent=1))
    print(len(runs), "records")


if __name__ == "__main__":
    main()
