"""Figure, LaTeX rows and summary for the 32B repeated compression-recovery
chains with the reference fixed to the base weights (run_base_chains_32b.py).

Inputs: results/probe_coverage_32b/run1/base_chains/*.json (one per method and
residual sparsity P, plus no_compression.json).  Every number is read from these
files.

For cycle c starting at step s = 200 + (c-1) R the chain is compressed and
restored at s ("immediate" evaluation) and trained R steps ("final" evaluation
at s + R).  Deltas are in 1e-3 nats:
  vs NC   - minus the no_compression chain at the same step (same data order)
  vs 200  - minus the uncompressed step-200 checkpoint

Outputs: paper/figs/gen/f1_base_chains_32b.{pdf,png,provenance.json},
paper/figs/gen/base_chains_rows.tex and results/.../base_chains/summary.md.

Run from code/checkpoint_compress/:
    python experiments/scripts/paper_figures/base_chains_fig.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import style as S  # noqa: E402
from make_figures import OUT, RUN32, zero_line  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (Agg backend set in style)

S.METHODS.update(
    {
        "first_order_ucb": S.MethodStyle("Uncertainty-protected $|gd|$ ($B{=}16$)", "#117733", "d", "--"),
        "iterative_ucb": S.MethodStyle("Ours: iterative + uncertainty ($K{=}4$, $B{=}4$)", S.C["vermillion"], "*"),
    }
)
ORDER = ["excp_style", "inshrinkerator_style", "dacp", "first_order_ucb", "iterative_ucb"]
PLAIN = {
    "excp_style": "ExCP-style (abs d)",
    "inshrinkerator_style": "Inshrinkerator-style (abs g*theta)",
    "dacp": "DACP (block HVP + Weibull)",
    "first_order_ucb": "first-order UCB one-shot (B=16)",
    "iterative_ucb": "ours: iterative UCB (K=4, B=4)",
}
TEX = {
    "excp_style": "ExCP-style",
    "inshrinkerator_style": "Inshrinkerator-style",
    "dacp": "DACP",
    "first_order_ucb": "Unc.-protected $|gd|$ ($B{=}16$)",
    "iterative_ucb": "Ours ($K{=}4$, $B{=}4$)",
}


def load(root: Path):
    recs, paths = {}, []
    nc = None
    for p in sorted(root.glob("*.json")):
        r = json.loads(p.read_text())
        if "cycles" not in r:
            continue
        paths.append(p)
        if r["method"] == "no_compression":
            nc = r
        else:
            recs[(r["method"], r["sparsity"])] = r
    return nc, recs, paths


def nc_val(nc, cycle: int, when: str, key: str = "val") -> float:
    return nc["cycles"][cycle - 1][when][key]["loss"]


def trajectory(r, nc=None, delta: bool = False):
    """(steps, vals) through pre-restore, immediate and final points."""
    xs, ys = [], []
    prev = r["uncompressed"]["val"]["loss"]
    for c in r["cycles"]:
        s, R = c["step"], c["recovery_chunks"][1] - c["recovery_chunks"][0]
        pts = [(s, prev), (s, c["immediate"]["val"]["loss"]), (s + R, c["final"]["val"]["loss"])]
        for x, y in pts:
            if delta:
                ref = nc_val(nc, c["cycle"], "immediate") if x == s else nc_val(nc, c["cycle"], "final")
                y = y - ref
            xs.append(x)
            ys.append(y * 1e3 if delta else y)
        prev = c["final"]["val"]["loss"]
    return xs, ys


def fmt(x: float) -> str:
    return f"{x:+.1f}"


def training_difference(r, nc):
    """Paired per-update losses, kept separate at each restoration boundary."""
    for cycle, reference in zip(r["cycles"], nc["cycles"]):
        logs, ref_logs = cycle["train_log"], reference["train_log"]
        steps = [point["step"] for point in logs]
        if steps != [point["step"] for point in ref_logs] or cycle["recovery_chunks"] != reference["recovery_chunks"]:
            raise ValueError("recovery training updates are not aligned")
        differences = [(point["loss"] - ref["loss"]) * 1e3 for point, ref in zip(logs, ref_logs)]
        yield steps, differences


def summary_md(nc, recs, sps: List[float]) -> str:
    u = nc["uncompressed"]["val"]["loss"]
    ug = nc["uncompressed"]["gate"]["loss"]
    C = len(nc["cycles"])
    L = ["# 32B repeated compression-recovery chains, reference = base weights", "",
         "Start: saved step-200 checkpoint (theta, AdamW moments), DeepSeek-R1-Distill-Qwen-32B. Each cycle "
         "compresses theta - base at residual sparsity P, restores pruned coordinates to base, keeps the moments, "
         f"evaluates, trains R = {nc['recovery_steps']} steps on the next recovery chunks (same order for all methods) "
         "and evaluates again. Deltas in 1e-3 nats. `vs NC` = minus the no_compression chain at the same step; "
         "`vs 200` = minus the uncompressed step-200 checkpoint "
         f"(val {u:.5f}, gate {ug:.5f}).", "",
         "## No-compression reference", "", "| step | val | gate |", "|---|---|---|"]
    L.append(f"| {nc['cycles'][0]['step']} | {u:.5f} | {ug:.5f} |")
    for c in nc["cycles"]:
        e = c["recovery_chunks"][1]
        L.append(f"| {e} | {c['final']['val']['loss']:.5f} | {c['final']['gate']['loss']:.5f} |")
    for sp in sps:
        ms = [m for m in ORDER if (m, sp) in recs]
        L += ["", f"## P = {sp}", "", "Validation loss delta vs NC (immediate / after recovery):", ""]
        L.append("| method | " + " | ".join(f"c{c} imm | c{c} fin" for c in range(1, C + 1)) + " |")
        L.append("|---|" + "---|" * (2 * C))
        for m in ms:
            r = recs[(m, sp)]
            cells = []
            for c in r["cycles"]:
                cells += [fmt((c["immediate"]["val"]["loss"] - nc_val(nc, c["cycle"], "immediate")) * 1e3),
                          fmt((c["final"]["val"]["loss"] - nc_val(nc, c["cycle"], "final")) * 1e3)]
            cells += ["n/a"] * (2 * C - len(cells))
            L.append(f"| {PLAIN[m]} | " + " | ".join(cells) + " |")
        L += ["", "Validation loss delta vs uncompressed step 200:", ""]
        L.append("| method | " + " | ".join(f"c{c} imm | c{c} fin" for c in range(1, C + 1)) + " |")
        L.append("|---|" + "---|" * (2 * C))
        rows = [("no compression", nc)] + [(PLAIN[m], recs[(m, sp)]) for m in ms]
        for name, r in rows:
            cells = []
            for c in r["cycles"]:
                cells += [fmt((c["immediate"]["val"]["loss"] - u) * 1e3), fmt((c["final"]["val"]["loss"] - u) * 1e3)]
            cells += ["n/a"] * (2 * C - len(cells))
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L += ["", "Gate loss delta vs NC (immediate / after recovery):", ""]
        L.append("| method | " + " | ".join(f"c{c} imm | c{c} fin" for c in range(1, C + 1)) + " |")
        L.append("|---|" + "---|" * (2 * C))
        for m in ms:
            r = recs[(m, sp)]
            cells = []
            for c in r["cycles"]:
                cells += [fmt((c["immediate"]["gate"]["loss"] - nc_val(nc, c["cycle"], "immediate", "gate")) * 1e3),
                          fmt((c["final"]["gate"]["loss"] - nc_val(nc, c["cycle"], "final", "gate")) * 1e3)]
            L.append(f"| {PLAIN[m]} | " + " | ".join(cells) + " |")
        L += ["", "Per-cycle cost and residual (cycle 1 / 2 / 3):", "",
              "| method | zero-residual frac. | residual L2 | scoring+selection s | stage s (ours) | score peak GiB (max dev) | recovery peak GiB | chosen payload GB |",
              "|---|---|---|---|---|---|---|---|"]
        for m in ms:
            r = recs[(m, sp)]
            cs = r["cycles"]
            z = " / ".join(f"{c['score']['residual']['zero_residual_fraction']:.3f}" for c in cs)
            l2 = " / ".join(f"{c['score']['residual']['residual_l2']:.2f}" for c in cs)
            ts = " / ".join(f"{c['score_total_s']:.0f}" for c in cs)
            st = " / ".join(",".join(f"{s['stage_s']:.0f}" for s in c["score"]["stages"]) for c in cs) \
                if m == "iterative_ucb" else "-"
            pk = " / ".join(f"{max(c['score_peak_gib'].values()):.1f}" for c in cs)
            rp = " / ".join(f"{max(c['recovery_peak_gib'].values()):.1f}" for c in cs)
            pay = " / ".join(f"{c['restore']['payload']['chosen'] / 1e9:.2f}" for c in cs)
            L.append(f"| {PLAIN[m]} | {z} | {l2} | {ts} | {st} | {pk} | {rp} | {pay} |")
    L += ["", "Train loss: per-step values are in each JSON (`cycles[].train_log`). "
          "Mean over the last 10 recovery steps of each cycle:", "",
          "| chain | " + " | ".join(f"c{c}" for c in range(1, C + 1)) + " |", "|---|" + "---|" * C]
    for name, r in [("no compression", nc)] + [(f"{PLAIN[m]} P={sp}", recs[(m, sp)]) for sp in sps for m in ORDER
                                               if (m, sp) in recs]:
        cells = [f"{sum(t['loss'] for t in c['train_log'][-10:]) / len(c['train_log'][-10:]):.4f}" for c in r["cycles"]]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "Lines between evaluation points in the figure are linear interpolations; only the marked points "
          "are measured."]
    return "\n".join(L) + "\n"


def tex_rows(nc, recs, sps: List[float]) -> str:
    rows = []
    C = len(nc["cycles"])
    for sp in sps:
        ms = [m for m in ORDER if (m, sp) in recs]
        vals = {m: [] for m in ms}
        for m in ms:
            for c in recs[(m, sp)]["cycles"]:
                vals[m] += [(c["immediate"]["val"]["loss"] - nc_val(nc, c["cycle"], "immediate")) * 1e3,
                            (c["final"]["val"]["loss"] - nc_val(nc, c["cycle"], "final")) * 1e3]
        n = 2 * C
        best = [min(vals[m][j] for m in ms if len(vals[m]) > j) for j in range(n)]
        head = f"\\multirow{{{len(ms)}}}{{*}}{{{sp:g}}}"
        for m in ms:
            cells = []
            for j in range(n):
                if j >= len(vals[m]):
                    cells.append("--")
                    continue
                t = fmt(vals[m][j])
                cells.append(f"$\\mathbf{{{t}}}$" if vals[m][j] == best[j] else f"${t}$")
            rows.append(f"{head} & {TEX[m]} & " + " & ".join(cells) + " \\\\")
            head = ""
        rows.append("\\midrule")
    return "\n".join(rows[:-1]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(RUN32 / "base_chains"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    nc, recs, paths = load(root)
    if nc is None:
        raise SystemExit("no_compression.json missing")
    sps = sorted({sp for _, sp in recs})
    (root / "summary.md").write_text(summary_md(nc, recs, sps), encoding="utf-8")
    (out / "base_chains_rows.tex").write_text(tex_rows(nc, recs, sps), encoding="utf-8")

    S.setup()
    fig, axes = plt.subplots(2, len(sps), figsize=(S.DOUBLE_W, 4.7), squeeze=False, sharex=True)
    restore_steps = [c["step"] for c in nc["cycles"]]
    for j, sp in enumerate(sps):
        top, bot = axes[0][j], axes[1][j]
        xs, ys = trajectory(nc)
        top.plot(xs, ys, **S.line_kw("no_compression", marker=False, ls="--"))
        top.plot(xs, ys, ls="", marker="o", ms=2.2, color="k")
        for m in ORDER:
            if (m, sp) not in recs:
                continue
            xs, ys = trajectory(recs[(m, sp)])
            top.plot(xs, ys, **S.line_kw(m, ls="--"))
            for steps, differences in training_difference(recs[(m, sp)], nc):
                bot.plot(steps, differences, color=S.style_of(m).color, lw=0.45, alpha=0.25)
                bot.plot(steps, S.moving_average(differences, 10), **S.line_kw(m, marker=False))
        for ax in (top, bot):
            for s in restore_steps:
                ax.axvline(s, color="0.8", lw=0.5, ls=":", zorder=0)
        zero_line(bot)
        top.set_title(f"$P={sp:g}$ ({1 / (1 - sp):.0f}$\\times$): validation endpoints")
        bot.set_title("Per-update training difference; trailing 10-step mean")
        bot.set_xlabel("Training step")
        if j == 0:
            top.set_ylabel("Validation loss (nats)")
            bot.set_ylabel(r"Training $\Delta$ vs. no compression ($10^{-3}$)")
    h, lab = axes[0][0].get_legend_handles_labels()
    keep = [i for i, l in enumerate(lab) if not l.startswith("_")]
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(h_pad=0.6, rect=(0, 0.09, 1, 1))
    fig.legend([h[i] for i in keep], [lab[i] for i in keep], loc="lower center", ncol=3, fontsize=6.5,
               frameon=False)
    S.save(fig, out, "f1_base_chains_32b", paths,
           notes={"validation_lines": "dashed linear guides between measured endpoints; no intermediate validation was recorded",
                  "training_lines": "raw paired per-update differences (faint) and trailing 10-step means; window resets each cycle",
                  "sparsities": sps, "methods": [m for m in ORDER if any(k[0] == m for k in recs)]})
    print(len(recs), "compressed chains; sparsities", sps)


if __name__ == "__main__":
    main()
