"""Validation loss of the study B long runs next to the no-compression run of the same seed.

Inputs: results/a30_long_20261121/runs/<setting>_s<seed>_{excp,dacpt}/ and BC_s<seed>_nocomp/.
Only seeds whose three runs reached the last step are used; curves are seed means, the band in
the gap panel spans the seed minimum and maximum.  Points: the loss before every checkpoint is
written, the reconstructed model at each resumption step, and 10/25/50/100/200 steps after it.
The noise control (BC_s42_noise, 0.1% random deletion, to step 2750) is drawn in the gap panel.

Panels: (a) loss over the whole run, (b) the last 2,500 steps, (c) gap to no compression in
1e-3 nats with the matching perplexity increase on the right axis.

Outputs: results/a30_long_20261121/reports/overlap/study_b_overlap_<setting>.{pdf,png,provenance.json}

Run from code/checkpoint_compress/:
    python experiments/scripts/paper_figures/study_b_overlap_fig.py --setting BO
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = HERE.parents[2] / "results" / "a30_long_20261121"
METHODS = [  # run suffix, label, colour, line style, z-order
    ("nocomp", "No compression", S.C["black"], "-", 3),
    ("excp", "ExCP", S.C["orange"], "-", 2),
    ("dacpt", r"DACP-G$_\rho$", S.C["blue"], "-", 4),
]
TITLES = {"BO": "ExCP threshold, equal deletion counts", "BC": r"$P=88\%$"}


def rows(run: Path):
    return [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines() if x.strip()]


def loss_series(run: Path):
    """(step, loss) in run order; a resumption step gives the pre-restore and the restored point."""
    out = []
    for r in rows(run):
        if r["kind"] == "checkpoint":
            out.append((r["step"], r["pre_loss"]))
            if r.get("break"):
                out.append((r["step"], r["post_loss"]))
        elif r["kind"] == "eval" and r["step"] > 0:
            out.append((r["step"], r["loss"]))
    return out


def base_lookup(run: Path):
    ck = {r["step"]: r["pre_loss"] for r in rows(run) if r["kind"] == "checkpoint"}
    ev = {r["step"]: r["loss"] for r in rows(run) if r["kind"] == "eval"}
    return ck, ev


def gap_series(run: Path, base: Path):
    ck, ev = base_lookup(base)
    out = []
    for r in rows(run):
        s = r["step"]
        if r["kind"] == "checkpoint" and s in ck:
            out.append((s, 1e3 * (r["pre_loss"] - ck[s])))
            if r.get("break"):
                out.append((s, 1e3 * (r["post_loss"] - ck[s])))
        elif r["kind"] == "eval" and s > 0 and s in ev:
            out.append((s, 1e3 * (r["loss"] - ev[s])))
    return out


def mean_band(curves):
    n = min(len(c) for c in curves)
    xs = [c[0] for c in curves[0][:n]]
    vals = [[c[i][1] for c in curves] for i in range(n)]
    return xs, [sum(v) / len(v) for v in vals], [min(v) for v in vals], [max(v) for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--setting", default="BO", choices=sorted(TITLES))
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "overlap")
    args = ap.parse_args()
    runs = args.root / "runs"

    seeds, inputs = [], []
    for base in sorted(runs.glob("BC_s*_nocomp")):
        seed = base.name.split("_s")[1].split("_")[0]
        cand = [base] + [runs / f"{args.setting}_s{seed}_{m}" for m, *_ in METHODS[1:]]
        if not all((d / "DONE").exists() for d in cand):
            continue
        cfg = json.loads((cand[1] / "config.json").read_text())
        if loss_series(cand[1])[-1][0] != cfg["total_steps"]:
            continue  # --stop-at runs are left out
        seeds.append(seed)
        inputs += [d / "metrics.jsonl" for d in cand]
    if not seeds:
        sys.exit("no seed has all three runs finished")
    breaks = json.loads((runs / f"{args.setting}_s{seeds[0]}_excp" / "config.json").read_text())["break_steps"]

    S.setup()
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE_W, 2.15), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    ax_full, ax_zoom, ax_gap = axes

    for m, label, color, ls, z in METHODS:
        name = (lambda s: f"BC_s{s}_nocomp") if m == "nocomp" else (lambda s, m=m: f"{args.setting}_s{s}_{m}")
        xs, ys, _, _ = mean_band([loss_series(runs / name(s)) for s in seeds])
        for ax in (ax_full, ax_zoom):
            ax.plot(xs, ys, color=color, ls=ls, lw=0.9, label=label, zorder=z)
        if m == "nocomp":
            continue
        gx, gy, lo, hi = mean_band([gap_series(runs / name(s), runs / f"BC_s{s}_nocomp") for s in seeds])
        if len(seeds) > 1:
            ax_gap.fill_between(gx, lo, hi, color=color, alpha=0.18, lw=0)
        ax_gap.plot(gx, gy, color=color, ls=ls, lw=0.9, label=label, zorder=z)
        for x, y in zip(gx, gy):  # values clipped by the axis limit
            if y > 125:
                ax_gap.annotate(f"{y:.0f}", (x, 125), xytext=(2, -6), textcoords="offset points",
                                fontsize=5.5, color=color, ha="left", va="top")

    noise, base = runs / "BC_s42_noise", runs / "BC_s42_nocomp"
    if (noise / "metrics.jsonl").exists() and (base / "metrics.jsonl").exists():
        nx, ny = zip(*gap_series(noise, base))
        ax_gap.plot(nx, ny, color="0.45", lw=0.8, ls="--", label="0.1% random (seed 42)", zorder=1)
        inputs.append(noise / "metrics.jsonl")

    for ax in axes:
        for b in breaks:
            ax.axvline(b, color="0.78", lw=0.5, ls=":", zorder=0)
        ax.set_xlabel("Training step")
    ax_full.set_xlim(0, 7500)
    ax_full.set_ylim(3.35, 5.0)
    ax_full.set_xticks([0, 2500, 5000, 7500])
    ax_full.set_ylabel("Validation loss (nats)")
    ax_full.set_title("(a) Whole run", loc="left")
    ax_zoom.set_xlim(5000, 7500)
    ax_zoom.set_ylim(3.435, 3.53)
    ax_zoom.set_xticks([5000, 6250, 7500])
    ax_zoom.set_title("(b) Last 2,500 steps", loc="left")
    ax_gap.axhline(0.0, color="0.6", lw=0.6, zorder=0)
    ax_gap.set_xlim(0, 7500)
    ax_gap.set_ylim(-20, 125)
    ax_gap.set_xticks([0, 2500, 5000, 7500])
    ax_gap.set_ylabel(r"Gap ($10^{-3}$ nats)")
    ax_gap.set_title("(c) Gap to no compression", loc="left")
    sec = ax_gap.secondary_yaxis("right", functions=(lambda g: 100.0 * np.expm1(np.asarray(g) / 1e3),
                                                     lambda p: 1e3 * np.log1p(np.asarray(p) / 100.0)))
    sec.set_ylabel("PPL increase (%)")

    h, l = ax_full.get_legend_handles_labels()
    gh, gl = ax_gap.get_legend_handles_labels()
    h += [x for x, y in zip(gh, gl) if y not in l]
    l += [y for y in gl if y not in l]
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.8)
    notes = {"setting": args.setting, "title": TITLES[args.setting], "seeds": [int(s) for s in seeds]}
    paths = S.save(fig, args.out, f"study_b_overlap_{args.setting}", inputs, notes=notes)
    print(paths["png"], json.dumps(notes))


if __name__ == "__main__":
    main()
