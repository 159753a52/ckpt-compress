"""Loss gap of the study B long runs (160M pre-training, ExCP protocol) against training step.

Inputs: results/a30_long_20261121/runs/{BO,BC}_s<seed>_<method>/ and BC_s<seed>_nocomp/.
Gap = validation loss minus that of the no-compression run of the same seed at the same step
(1e-3 nats): before every checkpoint is written, right after the restoration at each resumption
step, and 10/25/50/100/200 steps after it.  Seeds whose runs reached the last step are averaged;
the band spans their minimum and maximum.

Outputs: results/a30_long_20261121/reports/paper/study_b_gap.{pdf,png,provenance.json}

Run from code/checkpoint_compress/:
    python experiments/scripts/paper_figures/study_b_fig.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

ROOT = HERE.parents[2] / "results" / "a30_long_20261121"
CURVES = [  # method id, label, colour, line style
    ("excp", "ExCP", S.C["orange"], "-"),
    ("inshrink", "Inshrinkerator", S.C["sky"], "-"),
    ("dacpf", r"DACP-G$_\rho$ ($4\times4$)", S.lighten(S.C["blue"], 0.45), "--"),
    ("dacpt", r"DACP-G$_\rho$", S.C["blue"], "-"),
]
PANELS = [("BO", r"(a) ExCP threshold, equal deletion counts"), ("BC", r"(b) $P=88\%$")]


def read(run: Path):
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines() if x.strip()]
    cfg = json.loads((run / "config.json").read_text())
    return rows, cfg


def series(run: Path, base: Path):
    """(step, gap) pairs in plotting order; a resumption step contributes its pre and post points."""
    rows, cfg = read(run)
    brows, _ = read(base)
    bev = {r["step"]: r["loss"] for r in brows if r["kind"] == "eval"}
    bck = {r["step"]: r["pre_loss"] for r in brows if r["kind"] == "checkpoint"}
    out = []
    for r in rows:
        s = r["step"]
        if r["kind"] == "checkpoint" and s in bck:
            out.append((s, 1e3 * (r["pre_loss"] - bck[s])))
            if r.get("break"):
                out.append((s, 1e3 * (r["post_loss"] - bck[s])))
        elif r["kind"] == "eval" and s > 0 and s in bev:
            out.append((s, 1e3 * (r["loss"] - bev[s])))
    return out, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "paper")
    args = ap.parse_args()
    runs = args.root / "runs"
    S.setup()
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE_W, 2.05), sharey=True)
    inputs, notes = [], {}
    for ax, (setting, title) in zip(axes, PANELS):
        breaks = set()
        for mid, label, color, ls in CURVES:
            curves = []
            for d in sorted(runs.glob(f"{setting}_s*_{mid}")):
                seed = d.name.split("_s")[1].split("_")[0]
                base = runs / f"BC_s{seed}_nocomp"
                if not ((d / "DONE").exists() and (base / "DONE").exists()):
                    continue
                ser, cfg = series(d, base)
                if not ser or ser[-1][0] != cfg["total_steps"]:
                    continue  # partial runs (--stop-at) are left out
                curves.append(ser)
                breaks |= set(cfg["break_steps"])
                inputs += [d / "metrics.jsonl", base / "metrics.jsonl"]
                notes.setdefault(setting, {}).setdefault(mid, []).append(int(seed))
            if not curves:
                continue
            n = min(len(c) for c in curves)
            xs = [p[0] for p in curves[0][:n]]
            ys = [sum(c[i][1] for c in curves) / len(curves) for i in range(n)]
            if len(curves) > 1:
                lo = [min(c[i][1] for c in curves) for i in range(n)]
                hi = [max(c[i][1] for c in curves) for i in range(n)]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
            ax.plot(xs, ys, color=color, ls=ls, lw=0.9, label=label)
        for b in sorted(breaks):
            ax.axvline(b, color="0.75", lw=0.5, ls=":", zorder=0)
        ax.axhline(0.0, color="0.6", lw=0.6, zorder=0)
        ax.set_yscale("symlog", linthresh=10, linscale=0.6)
        ax.yaxis.set_major_locator(FixedLocator([-10, 0, 10, 100, 1000]))
        ax.yaxis.set_minor_locator(FixedLocator([v * m for m in (10, 100) for v in range(2, 10)]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\N{MINUS SIGN}")))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-25, 800)
        ax.set_xlim(0, 7500)
        ax.set_xticks([0, 1250, 2500, 3750, 5000, 6250, 7500])
        ax.set_xlabel("Training step")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel(r"Loss gap ($10^{-3}$ nats)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.0)
    paths = S.save(fig, args.out, "study_b_gap", inputs, notes={"seeds": notes})
    print(paths["pdf"], json.dumps(notes))


if __name__ == "__main__":
    main()
