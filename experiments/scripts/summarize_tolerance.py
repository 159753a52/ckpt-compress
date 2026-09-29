"""Tolerance frontier summary: highest compression within a fixed quality tolerance.

Usage: python summarize_tolerance.py <tolerance_dir> [--out summary.md]

Inputs (every number is read from JSON):
  <dir>/tol-<regime>_seed<seed>.json       run_residual_sweep_small.py --prefix tol-
  <dir>/tol-best_<regime>_seed<seed>.json  run_residual_ablation_small.py --tasks best --prefix tol-
  <dir>/system_bytes/tol_<R>_*_s<seed>.json run_system_bytes_comparison.py (CPU)

Tolerances are fixed before looking at results: relative PPL increase 0.5%, 1%, 2%,
i.e. immediate-restoration loss increase dloss <= log(1 + tol) nats against the
uncompressed checkpoint, and system-level restored PPL <= (1 + tol) * checkpoint PPL.
All criteria use the seed mean.

Immediate level, per method (sparsity grid p_1 < ... < p_n):
  grid   = largest p_i such that every grid point p_j <= p_i satisfies the tolerance
           (a '*' marks methods where a larger grid point passes again after a failure)
  interp = linear interpolation of the seed-mean dloss between the last passing grid
           point and the first failing one; '<p_1' if p_1 already fails, '>=p_n' if all pass
  residual CR = 1 / (1 - P) on the eligible residual entries.
System level, per method and operating-point family: the max of fp32 bytes / seed-mean
zlib bytes over operating points whose seed-mean restored PPL is within the tolerance.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

TOLS = [0.005, 0.01, 0.02]
SWEEP_TAGS = ["excp_style", "inshrinkerator_style", "first_order", "dacp", "first_order_ucb"]
BEST_TAGS = ["iterative_first_order_K16_B16", "iterative_ucb_K16_B16"]
REGIME_LABEL = {"R1": "Pythia-410M R1 (step 1000 vs base)", "P160_R1": "Pythia-160M R1 (step 1000 vs base)",
                "G2_R1": "GPT-2 124M R1 (step 1000 vs base)"}


def mean_std(xs):
    m = statistics.fmean(xs)
    return m, (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def frontier(points, limit):
    """points: sorted [(p, value)], pass iff value <= limit. Returns (grid, interp, relapse)."""
    grid, interp, relapse = None, None, False
    for i, (p, v) in enumerate(points):
        if v <= limit:
            grid = p
            continue
        if i == 0:
            interp = "below"
        else:
            p0, v0 = points[i - 1]
            interp = p0 + (limit - v0) / (v - v0) * (p - p0)
        relapse = any(v2 <= limit for _, v2 in points[i + 1:])
        break
    else:
        interp = "above"
    return grid, interp, relapse


def fmt_sp(x, lo, hi):
    if x == "below":
        return f"<{lo:g}"
    if x == "above":
        return f">={hi:g}"
    return "-" if x is None else f"{x:.3f}"


def fmt_cr(x):
    return "-" if not isinstance(x, float) else f"{1 / (1 - x):.2f}x"


# ----------------------------------------------------------------- immediate level
def load_immediate(d: Path):
    by_regime = defaultdict(lambda: defaultdict(list))  # regime -> tag -> [(seed, evals, unc_val)]
    meta = {}
    for f in sorted(d.glob("tol-*_seed*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        r = rec["regime"]
        tags = BEST_TAGS if rec.get("task") == "best" else SWEEP_TAGS
        for t in tags:
            if t in rec["evals"]:
                by_regime[r][t].append((rec["seed"], rec["evals"][t], rec["uncompressed"]["val"]))
        meta.setdefault(r, {"uncompressed": rec["uncompressed"], "full_restore": rec["full_restore"],
                            "residual": rec["residual"], "model": rec["model"]})
    return by_regime, meta


def immediate_section(regime, tags, meta):
    unc = meta["uncompressed"]
    lines = [f"## {REGIME_LABEL.get(regime, regime)}", "",
             f"Uncompressed val loss {unc['val']:.5f} (PPL {unc['ppl']:.4f}); full restore to the reference "
             f"val loss {meta['full_restore']['val']:.5f} (PPL {meta['full_restore']['ppl']:.4f}); "
             f"{meta['residual']['eligible_coords']} eligible residual entries.", ""]
    order = [t for t in SWEEP_TAGS + BEST_TAGS if t in tags]
    sps = sorted({float(p) for t in order for _, ev, _ in tags[t] for p in ev})
    lo, hi = sps[0], sps[-1]
    means = {}
    for t in order:
        pts = []
        for p in sps:
            ds = [(ev[str(p)]["val"] - u) for _, ev, u in tags[t] if str(p) in ev]
            if len(ds) == len(tags[t]):
                pts.append((p, statistics.fmean(ds), ds))
        means[t] = pts
    head = "| method | seeds | " + " | ".join(f"P@{int(tol * 1000) / 10:g}% interp / grid | CR" for tol in TOLS) + " |"
    lines += ["### Max eligible sparsity within tolerance (immediate restoration, seed mean)", "",
              "Tolerance dloss <= log(1+tol): 0.5% -> 4.99e-3, 1% -> 9.95e-3, 2% -> 1.98e-2 nats. "
              "CR = residual compression 1/(1-P) at the interpolated P (grid P in brackets).", "",
              head, "|---|---|" + "---|---|" * len(TOLS)]
    for t in order:
        cells = []
        for tol in TOLS:
            grid, interp, relapse = frontier([(p, m) for p, m, _ in means[t]], math.log1p(tol))
            g = "-" if grid is None else f"{grid:g}" + ("*" if relapse else "")
            cr = fmt_cr(interp) + (f" [{fmt_cr(grid)}]" if grid is not None else "")
            cells.append(f"{fmt_sp(interp, lo, hi)} / {g} | {cr}")
        lines.append(f"| {t} | {len(tags[t])} | " + " | ".join(cells) + " |")
    lines += ["", "### dloss grid (x1e-3 nats, mean +- std over seeds; bold = lowest mean)", "",
              "| method | " + " | ".join(f"{p:g}" for p in sps) + " |", "|---" * (len(sps) + 1) + "|"]
    best = {p: min(m for t in order for q, m, _ in means[t] if q == p) for p in sps}
    for t in order:
        cells = []
        for p in sps:
            hit = [(m, ds) for q, m, ds in means[t] if q == p]
            if not hit:
                cells.append("-")
                continue
            m, sd = mean_std(hit[0][1])
            c = f"{m * 1e3:.1f} +- {sd * 1e3:.1f}"
            cells.append(f"**{c}**" if m == best[p] else c)
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return lines + [""]


# ----------------------------------------------------------------- system level
def load_system(d: Path):
    out = defaultdict(list)  # R -> records
    refs = {}
    for f in sorted(d.glob("tol_*_s*.json")):
        m = re.match(r"tol_(R\d)_", f.name)
        payload = json.loads(f.read_text(encoding="utf-8"))
        refs.setdefault(m.group(1), (payload["reference"], payload["provenance"].get("git_commit", "")))
        out[m.group(1)] += payload["records"]
    return out, refs


def system_section(R, records, ref):
    ref, commit = ref
    ckpt = ref["checkpoint"]["perplexity"]
    fp32 = ref["fp32_serialized_residual_and_moments"]["torch_save"]
    groups = defaultdict(list)
    for r in records:
        op = r["operating_point"]
        fam = "fixed" if "eligible_sparsity" in op else "alpha"
        key = (r["method"], fam, op["eligible_sparsity"] if fam == "fixed" else op["alpha"])
        groups[key].append(r)
    rows = []
    for (method, fam, x), recs in groups.items():
        ppl = [r["restored"]["perplexity"] for r in recs]
        z = [r["total_bytes"]["zlib6"] for r in recs]
        rows.append({"method": method, "fam": fam, "x": x, "n": len(recs),
                     "sp": statistics.fmean(r["eligible_sparsity"] for r in recs),
                     "ppl": mean_std(ppl), "zlib": statistics.fmean(z), "cr": fp32 / statistics.fmean(z)})
    rows.sort(key=lambda r: (r["method"], r["fam"], r["sp"]))
    lines = [f"## System level, Pythia-410M {R} (CPU, ExCP moment rule + 4-bit quantizer)", "",
             f"Checkpoint PPL {ckpt:.4f}; fp32 weights+exp_avg+exp_avg_sq {fp32 / 1e6:.1f} MB; code {commit[:7]}. "
             f"PPL limits: 0.5% {ckpt * 1.005:.4f}, 1% {ckpt * 1.01:.4f}, 2% {ckpt * 1.02:.4f}. "
             "Families: alpha = ExCP alpha grid (ours at ExCP's eligible sparsity); fixed = our selection "
             "at a fixed eligible sparsity, non-eligible tensors and moments by the ExCP rule at alpha 1e-2.", "",
             "### Max total CR within tolerance (seed-mean PPL; operating point and zlib MB in brackets)", "",
             "| method | family | " + " | ".join(f"{int(t * 1000) / 10:g}%" for t in TOLS) + " |",
             "|---|---|" + "---|" * len(TOLS)]
    fams = sorted({(r["method"], r["fam"]) for r in rows})
    for method, fam in fams + sorted({(m, "any") for m, _ in fams if sum(mm == m for mm, _ in fams) > 1}):
        cand = [r for r in rows if r["method"] == method and (fam == "any" or r["fam"] == fam)]
        cells = []
        for tol in TOLS:
            ok = [r for r in cand if r["ppl"][0] <= ckpt * (1 + tol)]
            if not ok:
                cells.append("none")
                continue
            b = max(ok, key=lambda r: r["cr"])
            label = f"a={b['x']:g}" if b["fam"] == "alpha" else f"P={b['x']:g}"
            cells.append(f"{b['cr']:.1f}x ({label}, P={b['sp']:.3f}, {b['zlib'] / 1e6:.1f} MB)")
        lines.append(f"| {method} | {fam} | " + " | ".join(cells) + " |")
    lines += ["", "### Operating points", "",
              "| method | family | point | n | eligible sparsity | restored PPL | rel. PPL | zlib MB | CR |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, sd = r["ppl"]
        lines.append(f"| {r['method']} | {r['fam']} | {r['x']:g} | {r['n']} | {r['sp']:.4f} | {m:.4f} +- {sd:.4f} | "
                     f"{(m / ckpt - 1) * 100:+.2f}% | {r['zlib'] / 1e6:.2f} | {r['cr']:.1f} |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    lines = ["# Compression at fixed quality tolerance", "",
             "Generated by `experiments/scripts/summarize_tolerance.py`; definitions in its docstring. "
             "Tolerances (0.5%, 1%, 2% relative PPL increase) were fixed before the runs.", ""]
    imm, meta = load_immediate(args.dir)
    for regime in [r for r in ("R1", "P160_R1", "G2_R1") if r in imm] + sorted(set(imm) - {"R1", "P160_R1", "G2_R1"}):
        lines += immediate_section(regime, imm[regime], meta[regime])
    sysrec, refs = load_system(args.dir / "system_bytes")
    for R in sorted(sysrec):
        lines += system_section(R, sysrec[R], refs[R])
    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
