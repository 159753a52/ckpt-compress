"""Figure and LaTeX rows for the Pythia-410M base-reference compression-recovery
chains (run_base_chains_small.py).

Inputs: results/small_base_chains/*_seed*.json.  Every number is read from these
files.  For every method, sparsity P and cycle c: held-out loss minus the
no_compression chain of the same seed at the same step (1e-3 nats), mean and
sample std over seeds, immediately after restoration and after the R recovery
steps.

Outputs: paper/figs/gen/f2_small_chains.{pdf,png,provenance.json} and
paper/figs/gen/small_chains_rows.tex (selected cycles).

Run from code/checkpoint_compress/:
    python experiments/scripts/paper_figures/small_chains_fig.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import style as S  # noqa: E402
from make_figures import OUT, REPO, zero_line  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

S.METHODS.update(
    {
        "first_order_ucb": S.MethodStyle("Uncertainty-protected $|gd|$ (one shot)", "#117733", "d", "--"),
        "iterative_ucb": S.MethodStyle("Ours: re-linearized + uncertainty", S.C["vermillion"], "*"),
    }
)
ORDER = ["excp_style", "inshrinkerator_style", "dacp", "first_order_ucb", "iterative_ucb"]
TEX = {
    "excp_style": "ExCP-style",
    "inshrinkerator_style": "Inshrinkerator-style",
    "dacp": "DACP",
    "first_order_ucb": "Unc.-protected $|gd|$",
    "iterative_ucb": "Ours",
}
TABLE_CYCLES = (1, 2, 5, 10)


def load(root: Path):
    runs, paths = defaultdict(dict), []
    for f in sorted(root.glob("*_seed*.json")):
        d = json.loads(f.read_text())
        runs[(d["method"], d["sparsity"])][d["seed"]] = d
        paths.append(f)
    return runs, paths


def gaps(res: dict, ref: dict, phase: str):
    R = res["recovery_steps"]
    ref_at = {c["step"]: c["immediate"]["val"]["loss"] for c in ref["cycles"]}
    ref_at.update({c["step"] + R: c["final"]["val"]["loss"] for c in ref["cycles"]})
    return [(c[phase]["val"]["loss"] - ref_at[c["step"] + (R if phase == "final" else 0)]) * 1e3
            for c in res["cycles"]]


def stats(runs, base, m, sp, phase):
    """cycle -> (mean, std, n) over seeds that have a no_compression chain."""
    per = [gaps(r, base[s], phase) for s, r in runs[(m, sp)].items() if s in base]
    n_cyc = min(len(v) for v in per)
    out = []
    for c in range(n_cyc):
        xs = [v[c] for v in per]
        out.append((statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0, len(xs)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results" / "small_base_chains"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    runs, paths = load(root)
    base = runs[("no_compression", None)]
    sps = sorted({sp for (m, sp) in runs if sp is not None})
    table = {(m, sp, ph): stats(runs, base, m, sp, ph) for sp in sps for m in ORDER if (m, sp) in runs
             for ph in ("immediate", "final")}

    rows = []
    for sp in sps:
        ms = [m for m in ORDER if (m, sp, "immediate") in table]
        cells = {m: [] for m in ms}
        for m in ms:
            for c in TABLE_CYCLES:
                for ph in ("immediate", "final"):
                    st = table[(m, sp, ph)]
                    cells[m].append(st[c - 1][0] if len(st) >= c else None)
        best = [min(cells[m][j] for m in ms if cells[m][j] is not None) for j in range(2 * len(TABLE_CYCLES))]
        head = f"\\multirow{{{len(ms)}}}{{*}}{{{sp:g}}}"
        for m in ms:
            cs = []
            for j, v in enumerate(cells[m]):
                if v is None:
                    cs.append("--")
                    continue
                t = f"{v:+.1f}"
                cs.append(f"$\\mathbf{{{t}}}$" if v == best[j] else f"${t}$")
            rows.append(f"{head} & {TEX[m]} & " + " & ".join(cs) + " \\\\")
            head = ""
        rows.append("\\midrule")
    (out / "small_chains_rows.tex").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    S.setup()
    fig, axes = plt.subplots(2, len(sps), figsize=(S.DOUBLE_W, 3.9), squeeze=False, sharex=True)
    for j, sp in enumerate(sps):
        for i, ph in enumerate(("immediate", "final")):
            ax = axes[i][j]
            for m in ORDER:
                if (m, sp, ph) not in table:
                    continue
                st = table[(m, sp, ph)]
                xs = list(range(1, len(st) + 1))
                mu = [s[0] for s in st]
                sd = [s[1] for s in st]
                ax.plot(xs, mu, **S.line_kw(m))
                ax.fill_between(xs, [a - b for a, b in zip(mu, sd)], [a + b for a, b in zip(mu, sd)],
                                color=S.style_of(m).color, alpha=0.15, lw=0)
            if ph == "immediate":
                ys = [s[0] - s[1] for m in ORDER if (m, sp, ph) in table for s in table[(m, sp, ph)]]
                if min(ys) > 0:
                    ax.set_yscale("log")
                else:
                    ax.set_yscale("symlog", linthresh=10)
                    zero_line(ax)
                ax.set_title(f"$P={sp:g}$ ({1 / (1 - sp):.0f}$\\times$ residual compression)")
            else:
                zero_line(ax)
            if j == 0:
                ax.set_ylabel("Restored" if ph == "immediate" else "After 50 steps")
            if i == 1:
                ax.set_xlabel("Compression\u2013recovery cycle")
    fig.supylabel(r"$\Delta$ loss vs. no compression ($10^{-3}$ nats)", fontsize=7.5)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(h_pad=0.5, rect=(0.02, 0.08, 1, 1))
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=6.5, frameon=False)
    seeds = {f"{m} {sp}": len(runs[(m, sp)]) for (m, sp) in runs}
    S.save(fig, out, "f2_small_chains", paths,
           notes={"band": "mean +- sample std over seeds", "seeds": seeds, "table_cycles": TABLE_CYCLES})
    for sp in sps:
        print(sp, {m: len(runs[(m, sp)]) for m in ORDER if (m, sp) in runs})


if __name__ == "__main__":
    main()
