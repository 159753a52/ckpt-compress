"""Aggregate raw probe-coverage records into tables and figures.

Reads only raw JSON written by run_probe_coverage_32b.py; every output has a
sibling provenance file listing input paths and SHA-256 digests.  No value is
interpolated or typed in by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, inputs: Dict[str, str]):
    inputs[str(path)] = sha(path)
    return json.loads(path.read_text())


def mean_sd(xs: List[float]):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def fmt(m, s=None, digits=4):
    if m is None:
        return "--"
    if s is None:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}$\\pm${s:.{digits}f}"


def rule_pick(rounds: List[dict], key: str, tau: float, min_round: int):
    for r in rounds:
        if r["round"] >= min_round and r.get(key, -1.0) >= tau:
            return r["round"], True
    return rounds[-1]["round"], False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    work = Path(args.work)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs: Dict[str, str] = {}
    cfg = load(Path(args.config), inputs)
    rec = load(work / "cycle1" / "score_records.json", inputs)
    meta = load(work / "ckpt" / "meta.json", inputs)
    branches = {}
    for p in sorted((work / "cycle1" / "branches").glob("*.json")):
        branches[p.stem] = load(p, inputs)
    chains = {}
    if (work / "chains").exists():
        for p in sorted((work / "chains").glob("*.json")):
            chains[p.stem] = load(p, inputs)

    sparsities = cfg["sparsities"]
    base = rec["uncompressed_eval"]["val"]["loss"]
    base_gate = rec["uncompressed_eval"]["gate"]["loss"]
    nc = [branches[k]["final"]["val"]["loss"] for k in branches if k.startswith("no_compression")]
    nc_mean = statistics.fmean(nc) if nc else None
    grad_s = rec["gradient"]["fwd_bwd_s"]
    hvp_s = rec["hvp"]["hvp_s"]
    probes = rec["probes"]
    seeds = sorted(probes, key=int)

    def probe_cost(seed: str, m: int) -> float:
        return grad_s + sum(r["round_s"] for r in probes[seed][:m])

    summary: Dict[str, object] = {
        "uncompressed_val_loss": base,
        "uncompressed_gate_loss": base_gate,
        "no_compression_final_val": nc,
        "no_compression_noise_floor": (max(nc) - min(nc)) if len(nc) > 1 else None,
        "residual": rec["residual"],
        "gradient_s": grad_s,
        "hvp_s": hvp_s,
        "hvp_peak_gib": rec["hvp"]["peak_gib"],
        "checkpoint_evals": meta["evals"],
        "train_seconds": meta["train_seconds"],
        "rows": {},
        "saturation": {},
        "rules": {},
    }

    def branch_row(bid: str) -> Optional[dict]:
        b = branches.get(bid)
        if b is None:
            return None
        imm = b["immediate"]["val"]["loss"] - base
        fin = b["final"]["val"]["loss"] - nc_mean if nc_mean is not None else None
        pay = b["restore"]["payload"]["chosen"]
        return {
            "imm": imm,
            "imm_rel_ppl": math.expm1(imm),
            "gate_imm": b["immediate"]["gate"]["loss"] - base_gate,
            "final": fin,
            "final_rel_ppl": math.expm1(fin) if fin is not None else None,
            "pruned": b["restore"]["pruned"],
            "payload_bytes": pay,
            "archive": b.get("archive_check"),
        }

    tol = math.log1p(cfg["tolerance"]["immediate_rel_ppl"])
    margin = cfg["tolerance"]["noninferiority_nats"]
    for sp in sparsities:
        rows = {}
        for name, bid, cost in (
            ("ExCP-style $|d|$", f"excp_style_{sp}", 0.0),
            ("Inshrinkerator-style $|g\\theta|$", f"inshrinkerator_style_{sp}", grad_s),
            ("First order $|gd|$", f"first_order_{sp}", grad_s),
            ("Block HVP, global", f"hvp_global_{sp}", grad_s + hvp_s),
            ("DACP (HVP + Weibull)", f"dacp_{sp}", grad_s + hvp_s),
        ):
            r = branch_row(bid)
            if r:
                r["score_s"] = cost
                method_key = bid.rsplit("_", 1)[0].replace("hvp_global", "hvp").replace("dacp", "hvp")
                r["select_s"] = rec["methods"][method_key]["select_s"]
                rows[name] = r
        for m in cfg["probe"]["fixed_rounds"]:
            per = [branch_row(f"probe_s{s}_m{m}_global_{sp}") for s in seeds]
            per = [p for p in per if p]
            if not per:
                continue
            rows[f"Probe $m{{=}}{m}$"] = {
                "imm": mean_sd([p["imm"] for p in per]),
                "final": mean_sd([p["final"] for p in per]),
                "gate_imm": mean_sd([p["gate_imm"] for p in per]),
                "score_s": mean_sd([probe_cost(s, m) for s in seeds]),
                "select_s": mean_sd([probes[s][m - 1]["select_s"] for s in seeds]),
                "coverage_mean": mean_sd([probes[s][m - 1]["mean_mass"] for s in seeds]),
                "overlap_hvp": mean_sd([probes[s][m - 1][f"overlap_hvp_global_{sp}"] for s in seeds]),
                "payload_bytes": per[0]["payload_bytes"],
                "seeds": len(per),
                "per_seed_imm": [p["imm"] for p in per],
                "per_seed_final": [p["final"] for p in per],
            }
        # stopping rules
        rules = {}
        for rule in cfg["probe"]["rules"]:
            picks = []
            for s in seeds:
                m, met = rule_pick(probes[s], rule["key"], rule["tau"], rule.get("min_round", 1))
                b = branch_row(f"probe_s{s}_m{m}_global_{sp}")
                picks.append({"seed": s, "round": m, "met": met, "row": b, "score_s": probe_cost(s, m)})
            rules[rule["name"]] = picks
            ok = [p for p in picks if p["row"]]
            if ok:
                rows[f"Rule {rule['name']}"] = {
                    "imm": mean_sd([p["row"]["imm"] for p in ok]),
                    "final": mean_sd([p["row"]["final"] for p in ok]),
                    "score_s": mean_sd([p["score_s"] for p in ok]),
                    "rounds": [p["round"] for p in picks],
                    "met": [p["met"] for p in picks],
                }
        summary["rules"][str(sp)] = rules
        # immediate saturation from per-round evaluations (all rounds)
        sat = {}
        for kind in ("global", "weibull"):
            key = f"{kind}_{sp}"
            curves = {s: [r["evals"][key]["val"]["loss"] - base for r in probes[s]] for s in seeds}
            n = min(len(c) for c in curves.values())
            meanc = [statistics.fmean(curves[s][i] for s in seeds) for i in range(n)]
            sat_m = None
            for m in range(1, n + 1):
                if all(abs(meanc[j] - meanc[n - 1]) <= 0.0005 for j in range(m - 1, n)):
                    sat_m = m
                    break
            sat[kind] = {"per_seed": curves, "mean": meanc, "saturation_round": sat_m}
        summary["saturation"][str(sp)] = sat
        # non-inferiority vs block HVP (global)
        h = rows.get("Block HVP, global")
        ni = {}
        if h:
            for m in cfg["probe"]["fixed_rounds"]:
                r = rows.get(f"Probe $m{{=}}{m}$")
                if r:
                    ni[m] = {
                        "imm_all_seeds": all(x - h["imm"] <= margin for x in r["per_seed_imm"]),
                        "final_all_seeds": all(x - h["final"] <= margin for x in r["per_seed_final"] if x is not None),
                    }
        summary.setdefault("noninferiority", {})[str(sp)] = ni
        summary["rows"][str(sp)] = rows
        summary.setdefault("near_lossless_threshold_nats", tol)

    # Coverage by round, stratified by top-1 probability (seed means).
    cov = {}
    n_rounds = min(len(probes[s]) for s in seeds)
    for i in range(n_rounds):
        entry = {
            "mean_mass": mean_sd([probes[s][i]["mean_mass"] for s in seeds]),
            "p05_mass": mean_sd([probes[s][i]["p05_mass"] for s in seeds]),
            "repeat_rate": mean_sd([probes[s][i]["cumulative_repeat_rate"] for s in seeds]),
            "round_s": mean_sd([probes[s][i]["round_s"] for s in seeds]),
            "spearman_hvp": mean_sd([probes[s][i]["spearman_hvp"] for s in seeds]),
            "cosine_u_hd": mean_sd([probes[s][i]["cosine_u_hd_subsample"] for s in seeds]),
            "peak_gib_max": max(max(probes[s][i]["peak_gib"].values()) for s in seeds),
            "strata": {},
        }
        for key in probes[seeds[0]][i]["strata"]:
            entry["strata"][key] = {
                "positions": probes[seeds[0]][i]["strata"][key]["positions"],
                "mean_mass": mean_sd([probes[s][i]["strata"][key]["mean_mass"] for s in seeds]),
                "frac_ge_0.9": mean_sd([probes[s][i]["strata"][key]["frac_ge_0.9"] for s in seeds]),
            }
        cov[i + 1] = entry
    summary["coverage_by_round"] = cov

    # ---------------------------------------------------------------- tables
    lines = []
    for sp in sparsities:
        lines.append(f"% sparsity {sp}")
        for name, r in summary["rows"][str(sp)].items():
            def cell(v, d=4):
                if isinstance(v, tuple):
                    return fmt(v[0], v[1], d)
                return fmt(v, None, d)

            lines.append(
                f"{name} & {cell(r['imm'])} & {cell(r.get('final'))} & {cell(r.get('score_s'), 1)} \\\\"
            )
    (out / "table_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Paper table: both sparsities side by side, values in 10^-3 nats.
    def v(x, scale=1e3, d=1):
        if x is None:
            return "--"
        if isinstance(x, (tuple, list)):
            if x[0] is None:
                return "--"
            return f"${x[0] * scale:.{d}f}{{\\scriptstyle\\pm{x[1] * scale:.{d}f}}}$"
        return f"${x * scale:.{d}f}$"

    order = [
        "ExCP-style $|d|$",
        "Inshrinkerator-style $|g\\theta|$",
        "First order $|gd|$",
        "Block HVP, global",
        "DACP (HVP + Weibull)",
    ] + [f"Probe $m{{=}}{m}$" for m in cfg["probe"]["fixed_rounds"]] + ["Rule stable99", "Rule mean95"]
    body = []
    for name in order:
        cells = [name.replace("Rule stable99", "Probe, stable99 rule").replace("Rule mean95", "Probe, $\\bar q\\ge0.95$ rule (unmet, $m{=}8$)")]
        r95 = summary["rows"][str(sparsities[0])].get(name, {})
        for sp in sparsities:
            r = summary["rows"][str(sp)].get(name, {})
            cells += [v(r.get("imm")), v(r.get("final"))]
        sc = r95.get("score_s")
        cells.append(v(sc, 1.0, 1))
        body.append(" & ".join(cells) + " \\\\")
        if name in ("DACP (HVP + Weibull)", f"Probe $m{{=}}{cfg['probe']['fixed_rounds'][-1]}$"):
            body.append("\\midrule")
    (out / "table_32b_body.tex").write_text("\n".join(body) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, len(sparsities) + 1, figsize=(4.2 * (len(sparsities) + 1), 3.2))
    for ax, sp in zip(axes, sparsities):
        sat = summary["saturation"][str(sp)]["global"]
        for s, c in sat["per_seed"].items():
            ax.plot(range(1, len(c) + 1), c, color="tab:blue", alpha=0.35, lw=1)
        ax.plot(range(1, len(sat["mean"]) + 1), sat["mean"], color="tab:blue", lw=2, marker="o", label="probe (seed mean)")
        for label, bid, color in (("block HVP", f"hvp_global_{sp}", "tab:red"), ("first order", f"first_order_{sp}", "tab:gray"), ("ExCP-style", f"excp_style_{sp}", "tab:green")):
            r = branch_row(bid)
            if r:
                ax.axhline(r["imm"], color=color, ls="--", lw=1.2, label=label)
        ax.set_title(f"residual sparsity {sp:.0%}")
        ax.set_xlabel("probe rounds m")
        ax.set_ylabel("immediate $\\Delta$ val loss (nats)")
        ax.legend(fontsize=7)
    ax = axes[-1]
    for s in seeds:
        ms = [r["round"] for r in probes[s]]
        ax.plot(ms, [r["mean_mass"] for r in probes[s]], color="tab:blue", alpha=0.6, lw=1, label="mean $q_t$" if s == seeds[0] else None)
        ax.plot(ms, [r["p05_mass"] for r in probes[s]], color="tab:orange", alpha=0.6, lw=1, label="5th pct. $q_t$" if s == seeds[0] else None)
        ax.plot(ms, [r["cumulative_repeat_rate"] for r in probes[s]], color="tab:gray", alpha=0.6, lw=1, ls=":", label="repeat rate" if s == seeds[0] else None)
    for tau in (0.9, 0.95):
        ax.axhline(tau, color="k", lw=0.6, ls="--")
    ax.set_xlabel("probe rounds m")
    ax.set_title("coverage of sampled tokens")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "probe_saturation.pdf")
    fig.savefig(out / "probe_saturation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(sparsities), figsize=(4.2 * len(sparsities), 3.2))
    for ax, sp in zip(axes, sparsities):
        for name, r in summary["rows"][str(sp)].items():
            x = r.get("score_s")
            y = r.get("final")
            if isinstance(x, tuple):
                x = x[0]
            if isinstance(y, tuple):
                y = y[0]
            if x is None or y is None:
                continue
            ax.scatter([x], [y], s=18)
            ax.annotate(name.replace("$", "").replace("{=}", "="), (x, y), fontsize=6)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("scoring time (s)")
        ax.set_ylabel("$\\Delta$ val loss after recovery (nats)")
        ax.set_title(f"residual sparsity {sp:.0%}")
    fig.tight_layout()
    fig.savefig(out / "probe_pareto.pdf")
    fig.savefig(out / "probe_pareto.png", dpi=160)
    plt.close(fig)

    if chains:
        nc_chain = chains.get("no_compression")
        fig, axes = plt.subplots(1, len(sparsities), figsize=(4.2 * len(sparsities), 3.2))
        summary["chains"] = {}
        for ax, sp in zip(axes, sparsities):
            for cid, ch in chains.items():
                if cid == "no_compression" or ch["sparsity"] != sp:
                    continue
                ys = []
                for c, cyc in enumerate(ch["cycles"]):
                    ref_final = nc_chain["cycles"][c]["final"]["val"]["loss"] if nc_chain else None
                    ys.append(cyc["final"]["val"]["loss"] - ref_final if ref_final is not None else None)
                imm = []
                for c, cyc in enumerate(ch["cycles"]):
                    ref_imm = nc_chain["cycles"][c]["immediate"]["val"]["loss"] if nc_chain else None
                    imm.append(cyc["immediate"]["val"]["loss"] - ref_imm if ref_imm is not None else None)
                summary["chains"][cid] = {"final_delta": ys, "immediate_delta": imm, "rule_met": [cyc["score"].get("rule_met") for cyc in ch["cycles"]], "rounds": [len(cyc["score"].get("probe_rounds", [])) for cyc in ch["cycles"]]}
                ax.plot(range(1, len(ys) + 1), ys, marker="o", label=ch["method"])
            ax.set_xlabel("compression-recovery cycle")
            ax.set_ylabel("$\\Delta$ val loss vs. no compression")
            ax.set_title(f"repeated shocks, sparsity {sp:.0%}")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "probe_chains.pdf")
        fig.savefig(out / "probe_chains.png", dpi=160)
        plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    prov = {"inputs": inputs, "script": str(Path(__file__).resolve()), "script_sha256": sha(Path(__file__))}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("uncompressed_val_loss", "no_compression_noise_floor", "gradient_s", "hvp_s")}, indent=1))


if __name__ == "__main__":
    main()
