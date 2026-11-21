#!/usr/bin/env python
"""Fix the deletion rate and kappa of study B from its pilot, by a rule set before the pilot ran.

For every pilot failure step t and selection, gap = final loss after the short recovery
minus that of the uncompressed continuation from the same state.  For each deletion rate P,
kappa* is the DACP kappa with the lower mean gap.  P qualifies when, at every t,

  (a) DACP(kappa*) has a smaller gap than the ExCP rule, and
  (b) the DACP gap is at most half the gap of restarting from the reference itself.

adv(P) is the mean relative gap reduction of DACP over ExCP.  The chosen P is the largest
qualifying P with adv >= 0.2, else the qualifying P with the largest adv; without any
qualifying P the run falls back to P = 0.84, kappa = 1, and the decision says so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", default="375,1125")
    ap.add_argument("--rates", default="0.5,0.7,0.84,0.95")
    ap.add_argument("--kappas", default="1,3")
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",")]
    rates = [float(x) for x in args.rates.split(",")]
    kappas = [float(x) for x in args.kappas.split(",")]

    def final(name):
        return json.loads((args.pilot / name / "summary.json").read_text())["final_loss"]

    def damage(name):
        s = json.loads((args.pilot / name / "summary.json").read_text())
        return s["failures"][0]["immediate_damage"] if s["failures"] else None

    table, rows = {}, []
    for t in steps:
        base = final(f"t{t}_nocomp")
        table[t] = {"restart": final(f"t{t}_restart") - base}
        for p in rates:
            table[t][("excp", p)] = final(f"t{t}_excp_p{p}") - base
            for k in kappas:
                table[t][("dacp", p, k)] = final(f"t{t}_dacp_p{p}_k{k:g}") - base
    qualifying = []
    for p in rates:
        kbest = min(kappas, key=lambda k: sum(table[t][("dacp", p, k)] for t in steps))
        a = all(table[t][("dacp", p, kbest)] < table[t][("excp", p)] for t in steps)
        b = all(table[t][("dacp", p, kbest)] <= 0.5 * table[t]["restart"] for t in steps)
        adv = sum((table[t][("excp", p)] - table[t][("dacp", p, kbest)]) / max(abs(table[t][("excp", p)]), 1e-3)
                  for t in steps) / len(steps)
        row = {"sparsity": p, "kappa": kbest, "cond_a": a, "cond_b": b, "adv": adv,
               "gaps": {str(t): {"excp": table[t][("excp", p)], "dacp": table[t][("dacp", p, kbest)],
                                 "restart": table[t]["restart"],
                                 "dacp_immediate": damage(f"t{t}_dacp_p{p}_k{kbest:g}"),
                                 "excp_immediate": damage(f"t{t}_excp_p{p}")} for t in steps}}
        rows.append(row)
        if a and b:
            qualifying.append(row)
    strong = [r for r in qualifying if r["adv"] >= 0.2]
    if strong:
        pick, why = max(strong, key=lambda r: r["sparsity"]), "largest qualifying rate with adv >= 0.2"
    elif qualifying:
        pick, why = max(qualifying, key=lambda r: r["adv"]), "qualifying rate with the largest adv"
    else:
        pick, why = {"sparsity": 0.84, "kappa": 1.0}, "no qualifying rate: fallback"
    dec = {"sparsity": pick["sparsity"], "kappa": pick["kappa"], "reason": why, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dec, indent=1))
    print(json.dumps({k: dec[k] for k in ("sparsity", "kappa", "reason")}))


if __name__ == "__main__":
    main()
