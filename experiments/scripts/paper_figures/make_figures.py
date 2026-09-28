"""Publication figures for the residual-checkpoint compression results.

Figures (stems written to paper/figs/gen/):
  a1_sweep_quality_32b        delta validation loss vs residual sparsity, two regimes
  a2_taylor_fidelity_32b      Taylor predictions vs measured score-set change
  b1_restore_curves_32b       training loss after one restoration (95 %, 98 %)
  b2_chains_32b               three compress-restore cycles, per-cycle delta val
  b3_scoring_cost_32b         scoring time and peak GPU memory per method
  c1_size_quality_pythia410m  zlib bytes vs restored perplexity (Pareto)
  c2_prune_ratio_pythia410m   final perplexity change vs prune ratio

Every number is read from the result JSONs; a figure whose inputs are missing
is skipped with a message.  Each output has a sibling .provenance.json with
the SHA-256 of every input file and of the scripts in this directory.

    python experiments/scripts/paper_figures/make_figures.py            # all
    python experiments/scripts/paper_figures/make_figures.py --only a1 c1
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import style as S  # noqa: E402
from loaders import (  # noqa: E402
    load_branches,
    load_chains,
    load_jobs,
    load_sweep,
    load_system_bytes,
    peak_max,
)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (Agg backend set in style)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

REPO = HERE.parents[2]  # .../code/checkpoint_compress
PAPER = REPO.parents[1] / "paper"
RUN32 = REPO / "results" / "probe_coverage_32b" / "run1"
CAMPAIGN = REPO / "results" / "probe_pythia410m_campaign"
OUT = PAPER / "figs" / "gen"

MAIN_SWEEP = [
    "excp_style",
    "inshrinkerator_style",
    "first_order",
    "hvp_global",
    "dacp",
    "probe_sample_m1",
    "probe_topk_m1",
    "probe_headtail_m2",
]
SECONDARY_SWEEP = ["probe_sample_m4", "probe_topk_m2", "probe_topk_m4", "probe_topk_m8", "probe_headtail_m4"]
REGIME_TITLE = {
    "step100": "(a) 100-step residual (compression lowers loss)",
    "base": "(b) 200-step residual vs. base weights",
}


# ------------------------------------------------------------------ helpers
def sp_x(sp: float) -> float:
    """Horizontal position of a sparsity P: -log10(1 - P)."""
    return -math.log10(1.0 - sp)


def sp_axis(ax, sps: Sequence[float], cr_top: bool = True, label: str = "Residual sparsity $P$") -> None:
    ax.xaxis.set_major_locator(FixedLocator([sp_x(s) for s in sps]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{s:g}" for s in sps])
    lo, hi = sp_x(min(sps)), sp_x(max(sps))
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(label)
    if cr_top:
        top = ax.secondary_xaxis("top")
        top.xaxis.set_major_locator(FixedLocator([sp_x(s) for s in sps]))
        top.xaxis.set_minor_locator(NullLocator())
        top.set_xticklabels([f"{1 / (1 - s):.0f}$\\times$" for s in sps])
        top.tick_params(axis="x", labelsize=6.8, pad=1.2)
        top.set_xlabel("Residual compression $1/(1-P)$", fontsize=7.5, labelpad=2.5)


def zero_line(ax) -> None:
    ax.axhline(0.0, color="0.6", lw=0.6, zorder=0)


def handles(names: Sequence[str], marker: bool = True, labels: Dict[str, str] = None) -> List[Line2D]:
    labels = labels or {}
    out = []
    for n in names:
        kw = S.line_kw(n, marker=marker)
        if n in labels:
            kw["label"] = labels[n]
        out.append(Line2D([], [], **kw))
    return out


def skip(name: str, why: str) -> None:
    print(f"[skip] {name}: {why}")


def done(name: str, paths) -> None:
    print(f"[ok]   {name}: {paths['pdf'].name}, {paths['png'].name}")


def mean_sd(xs: Sequence[float]):
    xs = list(xs)
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


# ------------------------------------------------------------ A1: sweep quality
def fig_a1(out: Path) -> None:
    name = "a1_sweep_quality_32b"
    inputs: Set[Path] = set()
    sw = load_sweep(RUN32, inputs)
    if not sw:
        return skip(name, "no sweep/{step100,base}.json")
    regimes = [r for r in ("step100", "base") if r in sw]
    fig, axes = plt.subplots(1, len(regimes), figsize=(S.DOUBLE_W, 2.55), squeeze=False)
    for ax, reg in zip(axes[0], regimes):
        res = sw[reg]
        u = res["uncompressed"]["val"]
        sps = res["sparsities"]
        for tag in SECONDARY_SWEEP + MAIN_SWEEP:
            ev = res["evals"].get(tag)
            if not ev:
                continue
            pts = [(sp_x(sp), 1e3 * (ev[str(sp)]["val"] - u)) for sp in sps if str(sp) in ev]
            ax.plot(*zip(*pts), **S.line_kw(tag, marker=S.style_of(tag).main))
        zero_line(ax)
        sp_axis(ax, sps)
        ax.set_title(REGIME_TITLE.get(reg, reg), pad=3)
        ax.grid(axis="y")
    axes[0][0].set_ylabel(r"$\Delta$ val. loss ($10^{-3}$ nats)")
    hs = handles(MAIN_SWEEP)
    hs.append(Line2D([], [], color=S.lighten(S.C["rose"], 0.5), lw=0.8, label="Other probe variants"))
    fig.legend(handles=hs, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.5)
    done(name, S.save(fig, out, name, inputs, {"y": "evals[tag][P].val - uncompressed.val", "top_axis": "1/(1-P)"}))


# ------------------------------------------------------- A2: Taylor fidelity
def fig_a2(out: Path) -> None:
    name = "a2_taylor_fidelity_32b"
    inputs: Set[Path] = set()
    sw = load_sweep(RUN32, inputs)
    regimes = [r for r in ("step100", "base") if r in sw and sw[r].get("taylor")]
    if not regimes:
        return skip(name, "no sweep taylor records")
    fig, axes = plt.subplots(1, len(regimes), figsize=(S.DOUBLE_W, 2.45), squeeze=False)
    pred = [
        ("first_order", "First-order term", "0.55", "--", "^"),
        ("taylor2", "Second-order Taylor", "0.3", "-.", "s"),
        ("additive_attribution", "Additive attribution", "0.7", ":", "D"),
    ]
    for ax, reg in zip(axes[0], regimes):
        res = sw[reg]
        u = res["uncompressed"]["score"]
        sps = res["sparsities"]
        tay = res["taylor"]
        tsp = sorted(float(k) for k in tay)
        for key, lab, col, ls, mk in pred:
            ax.plot([sp_x(s) for s in tsp], [1e3 * tay[str(s)][key] for s in tsp], color=col, ls=ls,
                    marker=mk, ms=3.4, mfc="white", mec=col, mew=0.8, lw=1.0, label=lab)
        ev = res["evals"]["hvp_global"]
        meas = [(sp_x(s), 1e3 * (ev[str(s)]["score"] - u)) for s in sps if str(s) in ev]
        ax.plot(*zip(*meas), color="black", marker="o", ms=3.6, lw=1.2, label="Measured (block-HVP mask)")
        zero_line(ax)
        sp_axis(ax, sps)
        ax.set_title(REGIME_TITLE.get(reg, reg), pad=3)
        ax.grid(axis="y")
        # annotate the gap between Taylor-2 and measurement at the largest sparsity
        s_hi = str(tsp[-1])
        y_m, y_t = 1e3 * (ev[s_hi]["score"] - u), 1e3 * tay[s_hi]["taylor2"]
        txt = f"at $P={tsp[-1]:g}$: measured ${y_m:+.0f}$,\nsecond-order Taylor ${y_t:+.0f}$"
        ax.text(0.98, 0.5 if reg == "step100" else 0.4, txt, transform=ax.transAxes, ha="right", va="center",
                fontsize=6.8, color="0.2")
    axes[0][0].set_ylabel(r"$\Delta$ score-set loss ($10^{-3}$ nats)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    done(name, S.save(fig, out, name, inputs, {
        "measured": "evals.hvp_global[P].score - uncompressed.score",
        "predicted": "taylor[P].{first_order,taylor2,additive_attribution} (mask hvp_global_P)"}))


# ------------------------------------------------------ B1: restore curves
B1_METHODS = ["excp_style", "inshrinkerator_style", "dacp", "probe_s0_m1_global"]


def fig_b1(out: Path) -> None:
    name = "b1_restore_curves_32b"
    inputs: Set[Path] = set()
    sps = [0.95, 0.98]
    ids = ["no_compression_r0"] + [f"{m}_{sp}" for m in B1_METHODS for sp in sps]
    br = load_branches(RUN32, inputs, ids)
    if "no_compression_r0" not in br:
        return skip(name, "no_compression_r0 branch missing")
    fig, axes = plt.subplots(2, 2, figsize=(S.DOUBLE_W, 3.5), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [1, 1.15]})
    labels = {"probe_s0_m1_global": "Probe sampled $m{=}1$ (seed 0)"}
    nc_tl = br["no_compression_r0"]["train_log"]
    nc_loss = {r["step"]: r["loss"] for r in nc_tl}
    first = nc_tl[0]["step"]
    for j, sp in enumerate(sps):
        ax, axd = axes[0][j], axes[1][j]
        for bid in ["no_compression_r0"] + [f"{m}_{sp}" for m in B1_METHODS]:
            if bid not in br:
                continue
            tl = br[bid]["train_log"]
            x = [r["step"] for r in tl]
            y = [r["loss"] for r in tl]
            nc = bid == "no_compression_r0"
            col = S.style_of(bid).color
            ax.plot(x, y, color=col, lw=0.5, alpha=0.25, zorder=1)
            ax.plot(x, S.moving_average(y, 5), color=col, lw=1.0, ls="--" if nc else "-", zorder=4 if nc else 3)
            if nc:
                continue
            xd = [s for s in x if s in nc_loss]
            d = [1e3 * (r["loss"] - nc_loss[r["step"]]) for r in tl if r["step"] in nc_loss]
            axd.plot(xd, d, color=col, lw=0.5, alpha=0.3, zorder=1)
            axd.plot(xd, S.moving_average(d, 5), color=col, lw=1.1, zorder=3)
        zero_line(axd)
        for a in (ax, axd):
            a.axvline(first - 0.5, color="0.35", ls="--", lw=0.7)
            a.grid(axis="y")
        ax.text(first + 0.5, 0.97, "restore", transform=ax.get_xaxis_transform(), fontsize=6.8, va="top", color="0.3")
        ax.set_title(f"({'ab'[j]}) residual sparsity {sp:g} ({1 / (1 - sp):.0f}$\\times$)", pad=3)
        axd.set_xlabel("Training step (restored at step 200)")
    axes[0][0].set_ylabel("Training loss")
    axes[1][0].set_ylabel("$\\Delta$ vs. no compression\n($10^{-3}$ nats)")
    hs = [Line2D([], [], color="black", ls="--", lw=1.0, label="No compression")]
    hs += [Line2D([], [], color=S.style_of(m).color, lw=1.1, label=labels.get(m, S.style_of(m).label)) for m in B1_METHODS]
    fig.legend(handles=hs, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=1.0, h_pad=0.6)
    done(name, S.save(fig, out, name, inputs, {
        "top": "train_log[].loss; faint = raw, solid = trailing 5-step mean",
        "bottom": "train_log[].loss - no_compression_r0 train_log[].loss at the same step"}))


# ------------------------------------------------------------- B2: chains
CHAIN_ORDER = ["no_compression", "excp_style", "inshrinkerator_style", "hvp_global", "dacp", "probe_mean95"]


def fig_b2(out: Path) -> None:
    name = "b2_chains_32b"
    inputs: Set[Path] = set()
    ch = load_chains(RUN32, inputs)
    by_method = {c["method"]: c for c in ch.values()}
    if "no_compression" not in by_method:
        return skip(name, "no chains/no_compression.json")
    methods = [m for m in CHAIN_ORDER if m in by_method]
    sp = next(c["sparsity"] for m, c in by_method.items() if m != "no_compression")
    fig = plt.figure(figsize=(S.DOUBLE_W, 3.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    axd = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2 = fig.add_subplot(gs[:, 1])
    nc_loss = {r["step"]: r["loss"] for cy in by_method["no_compression"]["cycles"] for r in cy["train_log"]}
    for m in methods:
        col = S.style_of(m).color
        nc = m == "no_compression"
        for cy in by_method[m]["cycles"]:
            x = [r["step"] for r in cy["train_log"]]
            y = [r["loss"] for r in cy["train_log"]]
            ax.plot(x, y, color=col, lw=0.45, alpha=0.22, zorder=1)
            ax.plot(x, S.moving_average(y, 5), color=col, lw=1.0, ls="--" if nc else "-", zorder=4 if nc else 3)
            if nc:
                continue
            xd = [s for s in x if s in nc_loss]
            d = [1e3 * (r["loss"] - nc_loss[r["step"]]) for r in cy["train_log"] if r["step"] in nc_loss]
            axd.plot(xd, d, color=col, lw=0.45, alpha=0.25, zorder=1)
            axd.plot(xd, S.moving_average(d, 5), color=col, lw=1.1, zorder=3)
    zero_line(axd)
    for a in (ax, axd):
        for cy in by_method["no_compression"]["cycles"]:
            a.axvline(cy["step"] + 0.5, color="0.35", ls="--", lw=0.7, zorder=0)
        a.grid(axis="y")
    for cy in by_method["no_compression"]["cycles"]:
        axd.text(cy["step"] + 1.5, 0.04, f"restore {cy['cycle']}", transform=axd.get_xaxis_transform(),
                 fontsize=6.6, va="bottom", color="0.3")
    plt.setp(ax.get_xticklabels(), visible=False)
    axd.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    axd.set_ylabel("$\\Delta$ vs. no compression\n($10^{-3}$ nats)")
    ax.set_title(f"(a) Three compress–restore cycles, $P={sp:g}$", pad=3)

    ref = {cy["cycle"]: cy["final"]["val"]["loss"] for cy in by_method["no_compression"]["cycles"]}
    for m in methods:
        if m == "no_compression":
            continue
        pts = [(cy["cycle"], 1e3 * (cy["final"]["val"]["loss"] - ref[cy["cycle"]]))
               for cy in by_method[m]["cycles"] if cy["cycle"] in ref]
        ax2.plot(*zip(*pts), **S.line_kw(m, lw=1.0, ms=3.6))
    zero_line(ax2)
    ax2.set_xticks(sorted(ref))
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel(r"$\Delta$ final val. loss ($10^{-3}$ nats)")
    ax2.set_title("(b) vs. no compression", pad=3)
    ax2.grid(axis="y")
    hs = [Line2D([], [], color="black", ls="--", lw=1.0, label="No compression")]
    hs += handles([m for m in methods if m != "no_compression"])
    fig.legend(handles=hs, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=1.2, h_pad=0.6)
    done(name, S.save(fig, out, name, inputs, {"b": "cycles[c].final.val.loss - no_compression cycles[c].final.val.loss"}))


# --------------------------------------------------------- B3: scoring cost
COST_ORDER = ["excp_style", "inshrinkerator_style", "probe_mean95", "hvp_global", "dacp"]


def _curvature_s(score: dict):
    """Time of the curvature part of scoring, if the method has one."""
    if "hvp" in score:
        return score["hvp"]["hvp_s"] + score["hvp"]["cache_forward_s"]
    if "probe_rounds" in score:
        return sum(r["round_s"] for r in score["probe_rounds"])
    return None


def _score_peak(score: dict) -> float:
    peaks = [peak_max(score["gradient"]["peak_gib"])]
    if "hvp" in score:
        peaks.append(peak_max(score["hvp"]["peak_gib"]))
    for r in score.get("probe_rounds", []):
        peaks.append(peak_max(r["peak_gib"]))
    return max(peaks)


def fig_b3(out: Path) -> None:
    name = "b3_scoring_cost_32b"
    inputs: Set[Path] = set()
    ch = load_chains(RUN32, inputs)
    by_method = {c["method"]: c for c in ch.values() if c["method"] != "no_compression"}
    methods = [m for m in COST_ORDER if m in by_method]
    if not methods:
        return skip(name, "no compressed chains")
    rows = []
    for m in methods:
        cys = [cy for cy in by_method[m]["cycles"] if cy.get("score")]
        tot = [cy["score_total_s"] for cy in cys]
        curv = [_curvature_s(cy["score"]) for cy in cys]
        rows.append(dict(m=m, tot=tot, curv=None if curv[0] is None else statistics.fmean(curv),
                         peak=max(_score_peak(cy["score"]) for cy in cys)))
    cap = by_method[methods[0]]["gpu"]["devices"][0]["total_gib"]
    labels = {"probe_mean95": "Probe\n(8 rounds)", "inshrinkerator_style": "Inshrink.-\nstyle",
              "hvp_global": "Block HVP\n(global)", "excp_style": "ExCP-\nstyle", "dacp": "DACP"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.DOUBLE_W, 2.3))
    xs = range(len(rows))
    for i, r in zip(xs, rows):
        col = S.style_of(r["m"]).color
        mu = statistics.fmean(r["tot"])
        ax.bar(i, mu, width=0.62, color=col, edgecolor="black", lw=0.5, zorder=2)
        ax.errorbar(i, mu, yerr=[[mu - min(r["tot"])], [max(r["tot"]) - mu]], color="black", lw=0.7, capsize=2, zorder=4)
        if r["curv"]:
            ax.bar(i, r["curv"], width=0.62, facecolor="none", edgecolor="white", hatch="////", lw=0, zorder=3)
        ax.text(i, max(r["tot"]) * 1.12, f"{mu:.0f}", ha="center", va="bottom", fontsize=7)
        ax2.bar(i, r["peak"], width=0.62, color=col, edgecolor="black", lw=0.5, zorder=2)
        ax2.text(i, r["peak"] + 2, f"{r['peak']:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(5, 1500)
    ax.set_ylabel("Scoring time per cycle (s)")
    ax.set_title("(a) Scoring wall time (mean, min–max of 3 cycles)", pad=3)
    ax.legend(handles=[Patch(facecolor="0.55", edgecolor="white", hatch="////", label="curvature part (HVP / probe rounds)")],
              loc="upper left", fontsize=6.8)
    ax2.axhline(cap, color="0.35", ls="--", lw=0.7)
    ax2.text(len(rows) - 0.5, cap - 3, f"device memory {cap:g} GiB", ha="right", va="top", fontsize=6.8, color="0.3")
    ax2.set_ylim(0, cap * 1.08)
    ax2.set_ylabel("Peak GPU memory (GiB)")
    ax2.set_title("(b) Peak memory during scoring (max device)", pad=3)
    for a in (ax, ax2):
        a.set_xticks(list(xs))
        a.set_xticklabels([labels.get(r["m"], S.style_of(r["m"]).label) for r in rows], fontsize=6.8)
        a.grid(axis="y", which="major")
        a.set_axisbelow(True)
    fig.tight_layout(w_pad=1.5)
    done(name, S.save(fig, out, name, inputs, {
        "a": "chains cycles[].score_total_s; hatched = hvp.hvp_s + hvp.cache_forward_s or sum(probe_rounds[].round_s)",
        "b": "max over cycles and devices of score.gradient/hvp/probe_rounds peak_gib"}))


# ------------------------------------------------- C1: size-quality Pareto
C1_ALPHA = ["excp_pipeline", "dacp_excpq", "probe_topk_m1_exact_global_excpq"]
C1_LABEL = {
    "excp_pipeline": "ExCP pipeline",
    "dacp_excpq": "DACP + ExCP quant.",
    "probe_topk_m1_exact_global_excpq": "Probe top-$k$ $m{=}1$ + ExCP quant.",
    "inshrinkerator_like": "Inshrinkerator-like",
}


def fig_c1(out: Path) -> None:
    name = "c1_size_quality_pythia410m"
    inputs: Set[Path] = set()
    recs, ref = load_system_bytes(CAMPAIGN / "system_bytes", inputs)
    if not recs:
        return skip(name, "no system_bytes records")
    fp32_mb = ref["fp32_serialized_residual_and_moments"]["torch_save"] / 1e6
    ckpt_ppl = ref["checkpoint"]["perplexity"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.DOUBLE_W, 2.6))
    series = {}
    for m in C1_ALPHA:
        groups: Dict[float, list] = {}
        for r in recs:
            if r["method"] == m:
                groups.setdefault(r["operating_point"]["alpha"], []).append(r)
        pts = []
        for a in sorted(groups):
            rs = groups[a]
            mb, mb_sd = mean_sd([r["total_bytes"]["zlib6"] / 1e6 for r in rs])
            ppl, ppl_sd = mean_sd([r["restored"]["perplexity"] for r in rs])
            pts.append((a, mb, mb_sd, ppl, ppl_sd, len(rs)))
        series[m] = pts
    insh = sorted((r["total_bytes"]["zlib6"] / 1e6, r["restored"]["perplexity"], r["operating_point"]["n_bins"])
                  for r in recs if r["method"] == "inshrinkerator_like")
    zoom_max_alpha = 6e-4
    for a_i, axis in enumerate((ax, ax2)):
        for m, pts in series.items():
            if a_i == 1:
                pts = [p for p in pts if p[0] <= zoom_max_alpha]
            kw = S.line_kw(m, label=C1_LABEL[m], ms=3.6)
            axis.errorbar([p[1] for p in pts], [p[3] for p in pts], xerr=[p[2] for p in pts], yerr=[p[4] for p in pts],
                          elinewidth=0.6, capsize=1.5, **kw)
        if a_i == 0 and insh:
            axis.plot([p[0] for p in insh], [p[1] for p in insh],
                      **S.line_kw("inshrinkerator_like", label=C1_LABEL["inshrinkerator_like"], ls=":", ms=4))
            for k, (mb, ppl, nb) in enumerate(insh):
                off, ha = [((-4, 4), "right"), ((-3, 6), "right"), ((0, 6), "center")][min(k, 2)]
                axis.annotate(f"{nb} bins", (mb, ppl), xytext=off, textcoords="offset points", fontsize=6.5,
                              ha=ha, color="0.25")
        axis.axhline(ckpt_ppl, color="0.35", ls="--", lw=0.7, zorder=0)
        axis.set_xscale("log")
        axis.grid(which="major")
        axis.set_xlabel("Checkpoint size, zlib (MB, log)")
        top = axis.secondary_xaxis("top", functions=(lambda x: fp32_mb / np.maximum(x, 1e-9),
                                                     lambda c: fp32_mb / np.maximum(c, 1e-9)))
        top.set_xlabel("Compression ratio vs. fp32 weights + Adam moments ($\\times$)", fontsize=7.5, labelpad=2.5)
        top.tick_params(axis="x", which="both", labelsize=6.8, pad=1.2)
        xt, ct = ([100, 200, 500, 1000], [5, 10, 20, 50]) if a_i == 0 else ([180, 200, 220, 240, 260, 280], [18, 20, 22, 24, 26])
        axis.xaxis.set_major_locator(FixedLocator(xt))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xticklabels([f"{v:g}" for v in xt])
        top.xaxis.set_major_locator(FixedLocator(ct))
        top.xaxis.set_minor_locator(NullLocator())
        top.set_xticklabels([f"{v:g}" for v in ct])
    ax.text(0.99, ckpt_ppl - 0.02, f"checkpoint PPL {ckpt_ppl:.3f}", transform=ax.get_yaxis_transform(), ha="right",
            va="top", fontsize=6.8, color="0.3")
    zp = [p for p in series.get("excp_pipeline", []) if p[0] <= zoom_max_alpha]
    groups = []  # merge labels of points whose sizes differ by < 1 %
    for p in sorted(zp, key=lambda p: p[1]):
        if groups and abs(p[1] / groups[-1][-1][1] - 1) < 0.01:
            groups[-1].append(p)
        else:
            groups.append([p])
    for k, g in enumerate(groups):
        txt = "$\\alpha$=" + ", ".join(f"{p[0]:g}" for p in sorted(g))
        if k == 0:
            off, ha = (3, 5), "left"
        elif k == len(groups) - 1:
            off, ha = (-6, 5), "right"
        elif k == len(groups) - 2:
            off, ha = (4, -10), "left"
        else:
            off, ha = (0, 5), "center"
        ax2.annotate(txt, (g[-1][1], g[-1][3]), xytext=off, textcoords="offset points", fontsize=6.3,
                     ha=ha, color=S.style_of("excp_style").color)
    ax.set_ylabel("Restored perplexity")
    ax.set_title("(a) All operating points", pad=3)
    ax2.set_title(f"(b) Zoom: $\\alpha\\leq{zoom_max_alpha:g}$", pad=3)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
    done(name, S.save(fig, out, name, inputs, {
        "x": "total_bytes.zlib6 (mean +- sd over seeds)", "y": "restored.perplexity (mean +- sd)",
        "top_axis": "reference.fp32_serialized_residual_and_moments.torch_save / zlib6", "excluded": ["smoke.json"]}))


# ------------------------------------------- C2: final PPL vs prune ratio
C2_METHODS = ["excp_style", "inshrinkerator_style", "first_order_exact_global", "taylor_exact_global", "dacp",
              "probe_sample_m1_exact_global", "probe_topk_m1_exact_global", "probe_headtail_m2_exact_global"]


def fig_c2(out: Path) -> None:
    name = "c2_prune_ratio_pythia410m"
    inputs: Set[Path] = set()
    jobs = load_jobs(CAMPAIGN / "main_k1", inputs)
    if not jobs:
        return skip(name, "no main_k1 job records")
    ratios = sorted(jobs)
    fig, ax = plt.subplots(figsize=(S.SINGLE_W, 2.8))
    nc_sd = [mean_sd([r["final"]["perplexity"] for r in jobs[p]["no_compression"].values()])[1] for p in ratios]
    band = statistics.fmean(nc_sd) * 1e3
    ax.axhspan(-band, band, color="0.9", zorder=0, lw=0)
    ax.text(sp_x(ratios[0]) - 0.02, band * 0.93, "no-compression seed s.d.", fontsize=6.4, va="top", color="0.35")
    methods = [m for m in C2_METHODS if any(m in jobs[p] for p in ratios)]
    width = 0.028  # horizontal dodge between neighbouring methods (axis units)
    labels = {"first_order_exact_global": "First-order", "taylor_exact_global": "Taylor (block HVP)"}
    for k, m in enumerate(methods):
        off = (k - (len(methods) - 1) / 2) * width / 2
        xs, ys, es = [], [], []
        for p in ratios:
            base, rec = jobs[p].get("no_compression", {}), jobs[p].get(m, {})
            seeds = sorted(set(base) & set(rec))
            if not seeds:
                continue
            d = [1e3 * (rec[s]["final"]["perplexity"] - base[s]["final"]["perplexity"]) for s in seeds]
            mu, sd = mean_sd(d)
            xs.append(sp_x(p) + off)
            ys.append(mu)
            es.append(sd)
        kw = S.line_kw(m, lw=0.8, ms=3.0)
        if m in labels:
            kw["label"] = labels[m]
        ax.errorbar(xs, ys, yerr=es, elinewidth=0.5, capsize=1.2, **kw)
    zero_line(ax)
    sp_axis(ax, ratios, cr_top=False, label="Prune ratio")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - width, hi + width)
    ax.set_ylabel("$\\Delta$ final PPL vs. no\ncompression ($10^{-3}$)")
    ax.grid(axis="y")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=6.8, handlelength=1.6,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    done(name, S.save(fig, out, name, inputs, {
        "y": "paired (method - no_compression) final.perplexity over shared seeds, mean +- sd",
        "band": "+- mean over prune ratios of sd(no_compression final.perplexity)"}))


FIGS = {"a1": fig_a1, "a2": fig_a2, "b1": fig_b1, "b2": fig_b2, "b3": fig_b3, "c1": fig_c1, "c2": fig_c2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS), help="subset of figures (default: all)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    S.setup()
    for key in args.only or FIGS:
        FIGS[key](args.out)


if __name__ == "__main__":
    main()
