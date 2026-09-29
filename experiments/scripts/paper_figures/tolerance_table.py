"""Max residual sparsity (and compression 1/(1-P)) within a fixed quality tolerance.

Tolerance: immediate loss increase <= log(1 + tol), tol in {0.5%, 1%, 2%}, fixed
before the runs.  The frontier is the linear interpolation of the seed-mean
loss change between adjacent measured sparsities (first crossing).
Sources (results/): tolerance/tol-*_<reg>_seed*.json, tolerance_low/ (optional),
small_sweep_*/ sweep files for extra points; 32B from
probe_coverage_32b/run1/{tolerance_grid_base.json, sweep/base.json,
iterative_ucb/iu_base_ucb_B8_K8.json}.
"""
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results"
OUT = REPO.parents[1] / "paper" / "figs" / "gen"
TOLS = [0.005, 0.01, 0.02]
METHODS = [("excp_style", "ExCP-style $|d|$"), ("inshrinkerator_style", "Inshrinkerator-style $|g\\theta|$"),
           ("first_order", "First order $|gd|$"), ("dacp", "DACP"), ("first_order_ucb", "Uncertainty-protected"),
           ("iterative_ucb_K16_B16", "Re-linearized + uncertainty")]
SMALL = [("Pythia-160M", "P160_R1", "small_sweep_pythia160m"), ("GPT-2 124M", "G2_R1", "small_sweep_gpt2"),
         ("Pythia-410M", "R1", "small_sweep_pythia410m")]
used = set()


def frontier(points, tol):
    lim = 1e3 * math.log1p(tol)
    pts = sorted(points)
    if not pts or pts[0][1] > lim:
        return None, pts[0][0] if pts else None
    for (p0, d0), (p1, d1) in zip(pts, pts[1:]):
        if d1 > lim:
            return p0 + (lim - d0) / (d1 - d0) * (p1 - p0), None
    return pts[-1][0], None


def small_points(reg, sweep_dir):
    per = defaultdict(lambda: defaultdict(list))
    files = []
    for d in (RES / "tolerance", RES / "tolerance_low"):
        files += sorted(d.glob(f"tol-*{reg}_seed*.json")) if d.exists() else []
    files += sorted((RES / sweep_dir).glob(f"{reg}_seed*.json")) + sorted((RES / sweep_dir).glob(f"abl-best_{reg}_seed*.json"))
    for p in files:
        stem = p.name.split("_seed")[0]
        for pre in ("tol-best_", "abl-best_", "tol-"):
            if stem.startswith(pre):
                stem = stem[len(pre):]
                break
        if stem != reg:
            continue
        r = json.loads(p.read_text())
        used.add(p)
        for tag, ev in r["evals"].items():
            for sp, v in ev.items():
                per[tag][float(sp)].append((r["seed"], (v["val"] - r["uncompressed"]["val"]) * 1e3))
    out = {}
    for tag, bysp in per.items():
        pts = []
        for sp, xs in bysp.items():
            seeds = {}
            for s, x in xs:
                seeds.setdefault(s, x)
            if len(seeds) >= 3:
                pts.append((sp, statistics.fmean(seeds.values())))
        out[tag] = pts
    return out


def q32_points():
    run = RES / "probe_coverage_32b" / "run1"
    out = defaultdict(list)
    for p in (run / "tolerance_grid_base.json", run / "sweep" / "base.json"):
        r = json.loads(p.read_text())
        used.add(p)
        for tag, ev in r["evals"].items():
            for sp, v in ev.items():
                out[tag].append((float(sp), (v["val"] - r["uncompressed"]["val"]) * 1e3))
    p = run / "iterative_ucb" / "iu_base_ucb_B8_K8.json"
    r = json.loads(p.read_text())
    used.add(p)
    out["iterative_ucb_K16_B16"] = [(s["sparsity"], (s["val"] - r["uncompressed"]["val"]) * 1e3) for s in r["stages"]]
    rp = run / "scoring_rounds" / "scoring_rounds_base.json"
    rr = json.loads(rp.read_text())
    used.add(rp)
    ev = rr["by_b"]["32"]["evals"]["ucb"]
    out["first_order_ucb"] = [(float(s), (v["val"] - rr["uncompressed"]["val"]) * 1e3) for s, v in ev.items()]
    return {k: sorted(dict(v).items()) for k, v in out.items()}


def cell(pts, tol):
    if not pts:
        return "--", None
    f, below = frontier(pts, tol)
    if f is None:
        return f"$<{below * 100:.0f}\\%$", None
    return f"${f * 100:.1f}\\%$ ({1 / (1 - f):.0f}$\\times$)", f


def main() -> None:
    models = [(m, small_points(reg, d)) for m, reg, d in SMALL] + [("DeepSeek-R1-Distill-32B", q32_points())]
    lines, summary = [], {}
    for tag, label in METHODS:
        cells = []
        for m, pts in models:
            txt, f = cell(pts.get(tag, []), 0.01)
            cells.append((txt, f))
            summary.setdefault(m, {})[tag] = {str(t): frontier(pts.get(tag, []), t)[0] for t in TOLS}
        lines.append((label, cells))
    best = [max((c[j][1] for _, c in lines if c[j][1] is not None), default=None) for j in range(len(models))]
    rows = []
    for label, cells in lines:
        txt = [f"$\\mathbf{{{c[0].strip('$').split(' ')[0].strip('$')}}}$ {c[0].split(' ', 1)[1]}" if c[1] is not None and c[1] == best[j] and ' ' in c[0] else c[0] for j, c in enumerate(cells)]
        rows.append(label + " & " + " & ".join(txt) + " \\\\")
        if label == "DACP":
            rows.append("\\midrule")
    out = OUT / "tolerance_rows.tex"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    prov = {"inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(used)},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tolerance": "1% relative PPL (dloss <= log 1.01)", "frontiers": summary}
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1))
    print("\n".join(rows))
    for m, d in summary.items():
        print(m, {k: {t: (round(v, 4) if v else v) for t, v in x.items()} for k, x in d.items()})


if __name__ == "__main__":
    main()
