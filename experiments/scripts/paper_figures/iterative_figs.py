"""Figures and LaTeX table rows for iterative / uncertainty-protected residual pruning.

Every number is read from raw result JSON; nothing is typed in by hand.

Inputs (relative to code/checkpoint_compress/results/):
* small_sweep_pythia410m/R{1,2,3}_seed{42,43,44}.json      run_residual_sweep_small.py
* small_sweep_pythia410m/abl-{sched,combo,batches,best}_R*_seed*.json
                                                            run_residual_ablation_small.py
  (task/tag mapping as in summarize_residual_ablation_small.py)
* probe_coverage_32b/run1/sweep/base.json                   one-shot scorers
* probe_coverage_32b/run1/iterative/iterative_base.json     iterative |gd| (stages[])
* probe_coverage_32b/run1/scoring_rounds/scoring_rounds_base.json   by_b[B].evals.{mean,ucb}
* probe_coverage_32b/run1/iterative_ucb/*.json              optional, skipped if absent
* probe_trajectory_32b/traj_*/trajectory/step*_vs_*.json    trajectory (sweep schema)

Delta = (val - uncompressed.val) * 1e3 (1e-3 nats), each record against its own
uncompressed value; Pythia entries are mean (± s.d.) over the seeds present.

Outputs (paper/figs/gen/): e1_pythia_sweep, e2_pythia_ablation, e3_32b_rounds
(.pdf/.png/.provenance.json) and pythia_sweep_rows, pythia_ablation_rows,
q32_iterative_rows, q32_trajectory_rows (.tex + .provenance.json).

Run from code/checkpoint_compress/:
    python experiments/scripts/paper_figures/iterative_figs.py
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import style as S  # noqa: E402
from make_figures import OUT, REPO, RUN32, sp_axis, sp_x, zero_line  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (Agg backend set in style)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

SMALL = REPO / "results" / "small_sweep_pythia410m"
TRAJ = REPO / "results" / "probe_trajectory_32b"

# Styles for the iterative / uncertainty scorers (registered at runtime so that
# style.py, and with it the provenance of the older figures, stays unchanged).
S.METHODS.update(
    {
        "first_order_ucb": S.MethodStyle("Uncertainty-protected $|gd|$", "#117733", "d", "--"),
        "iterative_first_order": S.MethodStyle("Iterative $|gd|$", "#999933", "X"),
        "iterative_ucb_K16_B16": S.MethodStyle("Iterative + uncertainty ($K{=}16$, $B{=}16$)", S.C["black"], "*"),
    }
)

E1_METHODS = ["excp_style", "inshrinkerator_style", "first_order", "hvp_global", "dacp", "probe_topk_m1",
              "first_order_ucb", "iterative_first_order", "iterative_ucb_K16_B16"]
TAB_LABEL = {
    "excp_style": "ExCP-style $|d|$",
    "inshrinkerator_style": "Inshrinkerator-style $|g\\theta|$",
    "first_order": "First order $|gd|$",
    "hvp_global": "Block HVP",
    "dacp": "DACP",
    "probe_topk_m1": "Probe top-$k$ $m{=}1$",
    "first_order_ucb": "Uncertainty-protected $|gd|$",
    "iterative_first_order": "Iterative $|gd|$",
    "iterative_ucb_K16_B16": "Iterative + uncertainty, $K{=}16$, $B{=}16$",
}


# ------------------------------------------------------------------ helpers
def _load(path: Path) -> dict:
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    r["_path"] = Path(path).resolve()
    return r


def load_small() -> Dict[Tuple[str, str], List[dict]]:
    """{(task, regime): [records sorted by seed]}; task 'sweep' for R*_seed*.json."""
    out: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for p in sorted(SMALL.glob("R*_seed*.json")):
        r = _load(p)
        out[("sweep", r["regime"])].append(r)
    for p in sorted(SMALL.glob("abl-*_R*_seed*.json")):
        r = _load(p)
        out[(r["task"], r["regime"])].append(r)
    return {k: sorted(v, key=lambda r: r["seed"]) for k, v in out.items()}


def d_rec(rec: dict, tag: str, sp) -> Optional[float]:
    ev = rec["evals"].get(tag, {}).get(str(sp))
    return None if ev is None else (ev["val"] - rec["uncompressed"]["val"]) * 1e3


def stat(recs: Sequence[dict], tag: str, sp, used: Set[Path]):
    """(mean, sd, n) over seeds, or None unless every record has the cell."""
    xs = [d_rec(r, tag, sp) for r in recs]
    if not xs or any(x is None for x in xs):
        return None
    used.update(r["_path"] for r in recs)
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def stage_delta(rec: dict, sp: float) -> Optional[float]:
    for st in rec.get("stages", []):
        if abs(st["sparsity"] - sp) < 1e-9:
            return (st["val"] - rec["uncompressed"]["val"]) * 1e3
    return None


def write_rows(out: Path, stem: str, rows: list, inputs: Set[Path], notes: dict) -> List[str]:
    """rows: '\\midrule' or (label, [value or None, ...]). Bold = lowest per column."""
    data = [r for r in rows if not isinstance(r, str)]
    ncol = len(data[0][1])
    # compare at the printed precision so that displayed ties are all bold
    best = [min((round(r[1][j], 1) for r in data if r[1][j] is not None), default=None) for j in range(ncol)]
    lines = []
    for r in rows:
        if isinstance(r, str):
            lines.append(r)
            continue
        label, vals = r
        cells = []
        for j, v in enumerate(vals):
            if v is None:
                cells.append("--")
                continue
            t = f"{v:+.1f}"
            cells.append(f"$\\mathbf{{{t}}}$" if round(v, 1) == best[j] else f"${t}$")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    path = out / f"{stem}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prov = {
        "table_rows": stem,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": S.sha256(Path(__file__).resolve()),
        "inputs": {str(p): S.sha256(p) for p in sorted({Path(p).resolve() for p in inputs})},
        "output_sha256": S.sha256(path),
        "notes": notes,
    }
    (out / f"{stem}.provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"[ok]   {stem}.tex")
    for ln in lines:
        print("       " + ln)
    return lines


def log2_axis(ax, xs: Sequence[int], label: str) -> None:
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(list(xs)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(label)


def eb_kw(name: str, **over) -> dict:
    kw = S.line_kw(name)
    kw.update(capsize=1.6, elinewidth=0.6, capthick=0.6)
    kw.update(over)
    return kw


# -------------------------------------------------------- Pythia table rows
def pythia_sweep_rows(small, out: Path) -> None:
    used: Set[Path] = set()
    cols = [("R1", sp) for sp in ("0.9", "0.98", "0.995")] + [("R2", sp) for sp in ("0.9", "0.98", "0.995")]
    rows = []
    for tag in E1_METHODS:
        if tag == "first_order_ucb":
            rows.append("\\midrule")
        vals = []
        for reg, sp in cols:
            task = "best" if tag == "iterative_ucb_K16_B16" else "sweep"
            s = stat(small.get((task, reg), []), tag, sp, used)
            vals.append(None if s is None else s[0])
        rows.append((TAB_LABEL[tag], vals))
    write_rows(out, "pythia_sweep_rows", rows, used, {
        "columns": [f"{r} P={sp}" for r, sp in cols],
        "cell": "mean over seeds of (evals[tag][P].val - uncompressed.val) * 1e3",
        "sources": "R*_seed*.json (sweep); iterative_ucb_K16_B16 from abl-best_R*_seed*.json",
    })


def pythia_ablation_rows(small, out: Path) -> None:
    used: Set[Path] = set()
    sps = ["0.9", "0.95", "0.98", "0.99", "0.995"]

    def row(label, task, tag):
        return (label, [None if (s := stat(small.get((task, "R1"), []), tag, sp, used)) is None else s[0]
                        for sp in sps])

    rows = [row(f"Iterative, $K{{=}}{K}$", "sched", f"iterative_K{K}") for K in (1, 2, 4, 8, 16)]
    rows.append("\\midrule")
    fam = [("first_order", "$|gd|$"), ("first_order_ucb", "Uncertainty-protected $|gd|$"),
           ("iterative_first_order", "Iterative $|gd|$ (list)")]
    for tag, lab in fam:
        for B in (2, 16):
            rows.append(row(f"{lab}, $B{{=}}{B}$", "batches", f"{tag}_B{B}"))
    rows.append("\\midrule")
    rows += [row("Iterative Taylor (list)", "combo", "iterative_taylor"),
             row("Iterative probe (list)", "combo", "iterative_probe"),
             row("Iterative + uncertainty (list)", "combo", "iterative_ucb"),
             row("Iterative + uncertainty, $K{=}16$, $B{=}16$", "best", "iterative_ucb_K16_B16")]
    write_rows(out, "pythia_ablation_rows", rows, used, {
        "regime": "R1", "columns": [f"P={s}" for s in sps],
        "cell": "mean over seeds of (evals[tag][P].val - uncompressed.val) * 1e3; '--' = not run at P",
        "sources": "abl-sched (iterative_K*), abl-batches (*_B*), abl-combo (list-schedule scorers), abl-best",
    })


# ----------------------------------------------------------- E1: Pythia sweep
def fig_e1(small, out: Path) -> None:
    name = "e1_pythia_sweep"
    regimes = [r for r in ("R1", "R2", "R3") if ("sweep", r) in small]
    if not regimes:
        return print(f"[skip] {name}: no sweep records")
    used: Set[Path] = set()
    fig, axes = plt.subplots(1, len(regimes), figsize=(S.DOUBLE_W, 2.5), squeeze=False)
    for ax, reg in zip(axes[0], regimes):
        sw = small[("sweep", reg)]
        sps = sw[0]["sparsities"]
        for tag in E1_METHODS:
            recs = small.get(("best", reg), []) if tag == "iterative_ucb_K16_B16" else sw
            pts = [(sp_x(sp), *s[:2]) for sp in sps if (s := stat(recs, tag, sp, used)) is not None]
            if not pts:
                continue
            x, m, sd = zip(*pts)
            ax.errorbar(x, m, yerr=sd, **eb_kw(tag))
        ax.set_yscale("symlog", linthresh=10, linscale=0.6)
        zero_line(ax)
        sp_axis(ax, sps)
        ax.set_title(f"{reg} (n={len(sw)} seeds)", pad=3)
        ax.grid(axis="y")
    axes[0][0].set_ylabel(r"$\Delta$ eval loss ($10^{-3}$ nats, symlog)")
    hs = [Line2D([], [], **S.line_kw(t)) for t in E1_METHODS]
    fig.legend(handles=hs, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.0)
    S.save(fig, out, name, used, {
        "y": "mean ± s.d. over seeds of (evals[tag][P].val - uncompressed.val)*1e3; symlog, linthresh 10",
        "iterative_ucb_K16_B16": "from abl-best_<regime>_seed*.json, only where present",
    })
    print(f"[ok]   {name}")


# -------------------------------------------------------- E2: Pythia ablation
def fig_e2(small, out: Path) -> None:
    name = "e2_pythia_ablation"
    sched = small.get(("sched", "R1"), [])
    combo = small.get(("combo", "R1"), [])
    if not sched:
        return print(f"[skip] {name}: no abl-sched R1")
    used: Set[Path] = set()
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.DOUBLE_W, 2.45))
    Ks = [1, 2, 4, 8, 16]
    shades = {"0.95": S.lighten(S.C["green"], 0.6), "0.98": S.lighten(S.C["green"], 0.3), "0.995": S.C["green"]}
    for sp, col in shades.items():
        pts = [(K, *s[:2]) for K in Ks if (s := stat(sched, f"iterative_K{K}", sp, used)) is not None]
        if pts:
            x, m, sd = zip(*pts)
            a.errorbar(x, m, yerr=sd, color=col, marker="o", ms=3.5, lw=1.1, capsize=1.6, elinewidth=0.6,
                       label=f"$P={sp}$")
        lst = stat(combo, "iterative_first_order", sp, used)
        if lst:
            a.axhline(lst[0], color=col, ls="--", lw=0.8)
    a.plot([], [], color="0.4", ls="--", lw=0.8, label="list schedule")
    log2_axis(a, Ks, "Stages $K$ (geometric schedule)")
    a.set_ylabel(r"$\Delta$ eval loss ($10^{-3}$ nats)")
    a.set_title("(a) R1, iterative $|gd|$, $B{=}8$", pad=3)
    a.grid(axis="y")
    h, l = a.get_legend_handles_labels()
    order = [l.index(x) for x in ("$P=0.995$", "$P=0.98$", "$P=0.95$", "list schedule") if x in l]
    a.legend([h[i] for i in order], [l[i] for i in order], loc="upper right", ncol=2)

    Bs = [2, 4, 8, 16]
    sp = "0.98"
    for reg, ls, filled in (("R1", "-", True), ("R2", "--", False)):
        bt = small.get(("batches", reg), [])
        for fam in ("first_order", "first_order_ucb", "iterative_first_order"):
            pts = [(B, *s[:2]) for B in Bs if (s := stat(bt, f"{fam}_B{B}", sp, used)) is not None]
            if not pts:
                continue
            x, m, sd = zip(*pts)
            kw = eb_kw(fam, ls=ls, label=None)
            if not filled:
                kw.update(mfc="white")
            b.errorbar(x, m, yerr=sd, **kw)
    log2_axis(b, Bs, "Scoring batches $B$")
    b.set_title(f"(b) $P={sp}$", pad=3)
    b.grid(axis="y")
    hs = [Line2D([], [], **S.line_kw(f)) for f in ("first_order", "first_order_ucb", "iterative_first_order")]
    hs[0].set_label("One-shot $|gd|$")
    hs += [Line2D([], [], color="0.3", ls="-", marker="o", ms=3, label="R1"),
           Line2D([], [], color="0.3", ls="--", marker="o", ms=3, mfc="white", label="R2")]
    b.legend(handles=hs, loc="center right", bbox_to_anchor=(1.0, 0.54), ncol=2, fontsize=6.8)
    fig.tight_layout(w_pad=1.5)
    S.save(fig, out, name, used, {
        "a": "abl-sched R1 iterative_K{K}; dashed = abl-combo R1 iterative_first_order (list schedule)",
        "b": "abl-batches {R1,R2} {first_order,first_order_ucb,iterative_first_order}_B{B} at P=0.98",
        "y": "mean ± s.d. over seeds of (val - uncompressed.val)*1e3",
    })
    print(f"[ok]   {name}")


# ------------------------------------------------------------------- 32B
def load_32b():
    p = {"sweep": RUN32 / "sweep" / "base.json", "iter": RUN32 / "iterative" / "iterative_base.json",
         "rounds": RUN32 / "scoring_rounds" / "scoring_rounds_base.json"}
    d = {k: _load(v) for k, v in p.items() if v.exists()}
    iu = RUN32 / "iterative_ucb"
    d["iter_ucb"] = [_load(q) for q in sorted(iu.glob("*.json"))] if iu.exists() else []
    return d


def iter_ucb_label(r: dict) -> str:
    parts = [f"$B{{=}}{r['b']}$" if "b" in r else None, f"$K{{=}}{r['k']}$" if "k" in r else None]
    extra = ", ".join(x for x in parts if x)
    score = f" [{r['score']}]" if "score" in r else ""
    return f"Iterative + uncertainty{score}" + (f" ({extra})" if extra else "")


def q32_rows(d, out: Path) -> None:
    if not {"sweep", "rounds", "iter"} <= d.keys():
        return print("[skip] q32_iterative_rows: missing 32B inputs")
    sps = ["0.9", "0.95", "0.98", "0.99", "0.995"]
    used = {d[k]["_path"] for k in ("sweep", "rounds", "iter")}
    sw = d["sweep"]
    rows = [(lab, [d_rec(sw, tag, sp) for sp in sps]) for tag, lab in (
        ("excp_style", "ExCP-style"), ("inshrinkerator_style", "Inshrinkerator-style"),
        ("first_order", "First order ($B{=}4$)"), ("hvp_global", "Block HVP"), ("dacp", "DACP"))]
    rows.append("\\midrule")
    rd = d["rounds"]
    u = rd["uncompressed"]["val"]
    ev32 = rd["by_b"]["32"]["evals"]

    def rv(kind, sp):
        e = ev32[kind].get(sp)
        return None if e is None else (e["val"] - u) * 1e3

    rows.append(("Uncertainty-protected, $B{=}32$", [rv("ucb", sp) for sp in sps]))
    rows.append(("Mean $|gd|$, $B{=}32$", [rv("mean", sp) for sp in sps]))
    rows.append(("Iterative $|gd|$ ($B{=}4$)", [stage_delta(d["iter"], float(sp)) for sp in sps]))
    for r in d["iter_ucb"]:
        used.add(r["_path"])
        rows.append((iter_ucb_label(r), [stage_delta(r, float(sp)) for sp in sps]))
    write_rows(out, "q32_iterative_rows", rows, used, {
        "columns": [f"P={s}" for s in sps],
        "cell": "(val - uncompressed.val)*1e3, each file against its own uncompressed.val",
        "iterative_ucb_files": [str(r["_path"]) for r in d["iter_ucb"]],
    })


def fig_e3(d, out: Path) -> None:
    name = "e3_32b_rounds"
    if "rounds" not in d:
        return print(f"[skip] {name}: no scoring_rounds_base.json")
    rd = d["rounds"]
    u = rd["uncompressed"]["val"]
    Bs = sorted(int(b) for b in rd["by_b"])
    used = {rd["_path"]}
    fig, axes = plt.subplots(1, 2, figsize=(S.SINGLE_W, 2.6))
    refs = [("excp_style", "sweep"), ("hvp_global", "sweep"), ("iterative_first_order", "iter")]
    for ax, sp in zip(axes, ("0.98", "0.995")):
        for kind, tag in (("mean", "first_order"), ("ucb", "first_order_ucb")):
            pts = [(B, (rd["by_b"][str(B)]["evals"][kind][sp]["val"] - u) * 1e3) for B in Bs
                   if sp in rd["by_b"][str(B)]["evals"][kind]]
            ax.plot(*zip(*pts), **S.line_kw(tag, ls="-" if kind == "mean" else "--"))
        for tag, src in refs:
            if src not in d:
                continue
            v = d_rec(d["sweep"], tag, sp) if src == "sweep" else stage_delta(d["iter"], float(sp))
            if v is not None:
                used.add(d[src]["_path"])
                ax.axhline(v, color=S.style_of(tag).color, ls=":" if tag == "iterative_first_order" else (0, (4, 2)),
                           lw=0.9, zorder=1)
        for r in d["iter_ucb"]:
            v = stage_delta(r, float(sp))
            if v is not None:
                used.add(r["_path"])
                ax.axhline(v, color=S.style_of("iterative_ucb_K16_B16").color, ls=":", lw=0.9)
        log2_axis(ax, Bs, "Scoring batches $B$")
        ax.set_title(f"$P={sp}$", pad=3)
        ax.grid(axis="y")
        ax.tick_params(axis="x", labelsize=6.5)
    axes[0].set_ylabel(r"$\Delta$ val. loss ($10^{-3}$ nats)")
    hs = [Line2D([], [], **S.line_kw("first_order", ls="-", label="Mean $|gd|$")),
          Line2D([], [], **S.line_kw("first_order_ucb", label="Uncertainty-protected"))]
    hs += [Line2D([], [], color=S.style_of(t).color, lw=0.9,
                  ls=":" if t == "iterative_first_order" else (0, (4, 2)),
                  label={"excp_style": "ExCP-style", "hvp_global": "Block HVP",
                         "iterative_first_order": "Iterative $|gd|$ ($B{=}4$)"}[t]) for t, _ in refs]
    if d["iter_ucb"]:
        hs.append(Line2D([], [], color="k", ls=":", lw=0.9, label="Iterative + uncertainty"))
    fig.legend(handles=hs, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=0.8)
    S.save(fig, out, name, used, {
        "lines": "scoring_rounds_base.json by_b[B].evals.{mean,ucb}[P].val - uncompressed.val",
        "horizontal": "sweep/base.json excp_style, hvp_global (B=4 one-shot); iterative_base.json stages[P]",
    })
    print(f"[ok]   {name}")


# ------------------------------------------------------------- trajectory
DS_NAME = {"alpaca": "Alpaca", "wikitext": "WikiText"}
TRAJ_METHODS = ["excp_style", "inshrinkerator_style", "first_order", "hvp_global", "dacp", "probe_topk_m1"]


def q32_trajectory_rows(out: Path) -> None:
    runs = []
    for p in sorted(TRAJ.glob("traj_*/trajectory/step*_vs_*.json")):
        m = re.match(r"step(\d+)_vs_(\w+)$", p.stem)
        ref = m.group(2)
        ds = p.parts[-3].replace("traj_", "")
        # consecutive pairs by step, comparisons against base weights last
        runs.append(((ds, ref == "base", int(m.group(1))), ds, int(m.group(1)), ref, _load(p)))
    if not runs:
        return print("[skip] q32_trajectory_rows: no trajectory records")
    runs.sort(key=lambda x: x[0])
    sps = ["0.95", "0.98", "0.99"]
    lines, used = [], set()
    for i, (_, ds, step, ref, r) in enumerate(runs):
        used.add(r["_path"])
        if i:
            lines.append("\\midrule")
        base = r["uncompressed"]["val"]
        vals = {t: [(r["evals"][t][sp]["val"] - base) * 1e3 if sp in r["evals"].get(t, {}) else None for sp in sps]
                for t in TRAJ_METHODS if t in r["evals"]}
        best = [min((round(v[j], 1) for v in vals.values() if v[j] is not None), default=None)
                for j in range(len(sps))]
        head = f"{DS_NAME.get(ds, ds)} {step}/{ref.replace('step', '')}"
        for t, v in vals.items():
            cells = ["--" if x is None else (f"$\\mathbf{{{x:+.1f}}}$" if round(x, 1) == best[j] else f"${x:+.1f}$")
                     for j, x in enumerate(v)]
            lines.append(f"{head} & {TAB_LABEL[t]} & " + " & ".join(cells) + " \\\\")
            head = ""
    path = out / "q32_trajectory_rows.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prov = {
        "table_rows": "q32_trajectory_rows",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": S.sha256(Path(__file__).resolve()),
        "inputs": {str(p): S.sha256(p) for p in sorted(used)},
        "output_sha256": S.sha256(path),
        "notes": {"columns": [f"P={s}" for s in sps],
                  "cell": "(evals[tag][P].val - uncompressed.val)*1e3; bold = lowest within each checkpoint pair"},
    }
    (out / "q32_trajectory_rows.provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print("[ok]   q32_trajectory_rows.tex")
    for ln in lines:
        print("       " + ln)


def summary(small) -> None:
    """Print mean ± s.d. for the README (not written to any file)."""
    for reg in ("R1", "R2", "R3"):
        for tag in E1_METHODS:
            recs = small.get(("best", reg), []) if tag == "iterative_ucb_K16_B16" else small.get(("sweep", reg), [])
            cells = [f"{sp}:{s[0]:+.1f}±{s[1]:.1f}" for sp in ("0.9", "0.98", "0.995")
                     if (s := stat(recs, tag, sp, set())) is not None]
            print(f"  {reg} {tag}: " + " ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    S.setup()
    args.out.mkdir(parents=True, exist_ok=True)
    small = load_small()
    pythia_sweep_rows(small, args.out)
    pythia_ablation_rows(small, args.out)
    fig_e1(small, args.out)
    fig_e2(small, args.out)
    d = load_32b()
    q32_rows(d, args.out)
    fig_e3(d, args.out)
    q32_trajectory_rows(args.out)
    summary(small)


if __name__ == "__main__":
    main()
