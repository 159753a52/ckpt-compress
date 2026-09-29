"""Summarize run_base_chains_small.py results as Markdown.

For every sparsity, method and cycle: mean +- sample std over seeds of the
immediate and final held-out loss minus the no_compression chain of the same
seed at the same optimizer step (1e-3 nats), and per-cycle paired wins of
iterative_ucb against every baseline (seeds with strictly lower loss).

Usage: summarize_base_chains_small.py <results dir> [--out summary.md]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ORDER = ["excp_style", "inshrinkerator_style", "dacp", "first_order_ucb", "iterative_ucb"]
OURS = "iterative_ucb"


def load(root: Path) -> Dict:
    runs = defaultdict(dict)  # (method, sparsity) -> seed -> result
    for f in sorted(root.glob("*_seed*.json")):
        d = json.loads(f.read_text())
        runs[(d["method"], d["sparsity"])][d["seed"]] = d
    return runs


def ms(xs: List[float]) -> str:
    if not xs:
        return "-"
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{statistics.mean(xs):.1f} ± {sd:.1f}"


def gaps(res: dict, ref: dict, phase: str) -> List[float]:
    """Per-cycle loss minus the no_compression chain at the same step (1e-3 nats)."""
    ref_at = {c["step"]: c["immediate"]["val"]["loss"] for c in ref["cycles"]}
    ref_at.update({c["step"] + len(c["train_log"]["train_losses"]): c["final"]["val"]["loss"] for c in ref["cycles"]})
    out = []
    for c in res["cycles"]:
        step = c["step"] if phase == "immediate" else c["step"] + len(c["train_log"]["train_losses"])
        out.append((c[phase]["val"]["loss"] - ref_at[step]) * 1e3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    runs = load(args.root)
    base = runs.get(("no_compression", None), {})
    lines = ["# Pythia-410M base-reference compression-recovery chains", ""]
    any_run = next(iter(base.values()), None)
    if any_run:
        t = any_run["train"]
        lines += [f"Regime {any_run['regime']}, {any_run['cycles_n']} cycles x R = {any_run['recovery_steps']} "
                  f"recovery steps (AdamW moments kept, constant lr {t['lr']}, batch {t['batch_size']} x "
                  f"{t['seq_length']}). Values: held-out loss minus the no_compression chain of the same seed "
                  f"at the same step, 1e-3 nats, mean ± std over seeds. Uncompressed start loss "
                  f"{any_run['uncompressed']['val']['loss']:.5f}.", ""]
        nc = [[c["final"]["val"]["loss"] for c in r["cycles"]] for r in base.values()]
        lines += ["no_compression final loss per cycle (mean over seeds): "
                  + ", ".join(f"{statistics.mean(v):.4f}" for v in zip(*nc)), ""]
    for sp in sorted({s for (_, s) in runs if s is not None}):
        methods = [m for m in ORDER if (m, sp) in runs]
        lines += [f"## P = {sp}", ""]
        for phase in ("immediate", "final"):
            table = {}
            for m in methods:
                per_seed = {seed: gaps(r, base[seed], phase) for seed, r in runs[(m, sp)].items() if seed in base}
                table[m] = per_seed
            n_cyc = max((len(v) for t in table.values() for v in t.values()), default=0)
            lines += [f"### {phase} (seeds per method: "
                      + ", ".join(f"{m} {len(table[m])}" for m in methods) + ")", "",
                      "| cycle | " + " | ".join(methods) + " |", "|---" * (len(methods) + 1) + "|"]
            for c in range(n_cyc):
                row = [ms([v[c] for v in table[m].values() if len(v) > c]) for m in methods]
                lines.append(f"| {c + 1} | " + " | ".join(row) + " |")
            lines.append("")
            if OURS in table:
                others = [m for m in methods if m != OURS]
                lines += [f"Paired wins of {OURS} ({phase}, seeds with lower loss / paired seeds):", "",
                          "| cycle | " + " | ".join(others) + " |", "|---" * (len(others) + 1) + "|"]
                for c in range(n_cyc):
                    row = []
                    for m in others:
                        pair = [(table[OURS][s][c], table[m][s][c]) for s in table[OURS]
                                if s in table[m] and len(table[m][s]) > c and len(table[OURS][s]) > c]
                        row.append(f"{sum(a < b for a, b in pair)}/{len(pair)}" if pair else "-")
                    lines.append(f"| {c + 1} | " + " | ".join(row) + " |")
                lines.append("")
        tim = []
        for m in methods:
            sc = [c["score_total_s"] for r in runs[(m, sp)].values() for c in r["cycles"]]
            cy = [c["cycle_s"] for r in runs[(m, sp)].values() for c in r["cycles"]]
            pk = [r["peak_gib"] for r in runs[(m, sp)].values()]
            tim.append(f"| {m} | {statistics.mean(sc):.1f} | {statistics.mean(cy):.1f} | {max(pk):.2f} |")
        lines += ["Timing per cycle (mean over cycles and seeds):", "",
                  "| method | score s | cycle s | peak GiB |", "|---|---|---|---|", *tim, ""]
    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
