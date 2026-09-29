"""Cross-model table rows: immediate loss change at 98% and 99.5% residual sparsity.

Sources (results/): small_sweep_{pythia160m,gpt2,pythia410m}/ sweep files
``<regime>_seed*.json`` and ablation files ``abl-best_<regime>_seed*.json``;
32B from probe_coverage_32b/run1/{sweep/base.json, scoring_rounds/*.json,
iterative/iterative_base.json, iterative_ucb/iu_base_ucb_*.json (optional)}.
Cells: mean over seeds of (val - uncompressed.val) * 1e3; '--' when absent.
"""
import hashlib
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results"
OUT = REPO.parents[1] / "paper" / "figs" / "gen"
SPS = ["0.98", "0.995"]
SMALL = [("Pythia-160M", "small_sweep_pythia160m", "P160_R1"),
         ("GPT-2 124M", "small_sweep_gpt2", "G2_R1"),
         ("Pythia-410M", "small_sweep_pythia410m", "R1")]
ROWS = [("excp_style", "ExCP-style $|d|$"), ("inshrinkerator_style", "Inshrinkerator-style $|g\\theta|$"),
        ("first_order", "First order $|gd|$"), ("hvp_global", "Block HVP"), ("dacp", "DACP"),
        ("probe_topk_m1", "Probe top-$k$ $m{=}1$"), None,
        ("first_order_ucb", "Uncertainty-protected"), ("iterative_first_order", "Re-linearized"),
        ("iterative_ucb_K16_B16", "Re-linearized + uncertainty")]
used = set()


def small_cell(d, reg, tag, sp):
    pat = f"abl-best_{reg}_seed*.json" if tag.endswith("_K16_B16") else f"{reg}_seed*.json"
    xs = []
    for p in sorted((RES / d).glob(pat)):
        r = json.loads(p.read_text())
        ev = r["evals"].get(tag, {}).get(sp)
        if ev is None:
            continue
        used.add(p)
        xs.append((ev["val"] - r["uncompressed"]["val"]) * 1e3)
    return statistics.fmean(xs) if xs else None


def q32_cell(tag, sp):
    run = RES / "probe_coverage_32b" / "run1"
    if tag in ("excp_style", "inshrinkerator_style", "first_order", "hvp_global", "dacp", "probe_topk_m1"):
        p = run / "sweep" / "base.json"
        r = json.loads(p.read_text())
        used.add(p)
        return (r["evals"][tag][sp]["val"] - r["uncompressed"]["val"]) * 1e3
    if tag == "first_order_ucb":
        p = run / "scoring_rounds" / "scoring_rounds_base.json"
        r = json.loads(p.read_text())
        used.add(p)
        b = max(r["by_b"], key=int)
        return (r["by_b"][b]["evals"]["ucb"][sp]["val"] - r["uncompressed"]["val"]) * 1e3
    files = {"iterative_first_order": [run / "iterative" / "iterative_base.json"],
             "iterative_ucb_K16_B16": sorted((run / "iterative_ucb").glob("iu_base_ucb_*.json"))}[tag]
    for p in files:
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        for st in r["stages"]:
            if abs(st["sparsity"] - float(sp)) < 1e-9:
                used.add(p)
                return (st["val"] - r["uncompressed"]["val"]) * 1e3
    return None


def main() -> None:
    cols = [(m, d, reg, sp) for m, d, reg in SMALL for sp in SPS] + [("32B", None, None, sp) for sp in SPS]
    table = []
    for row in ROWS:
        if row is None:
            table.append("\\midrule")
            continue
        tag, label = row
        vals = [q32_cell(tag, sp) if m == "32B" else small_cell(d, reg, tag, sp) for m, d, reg, sp in cols]
        table.append((label, vals))
    data = [r for r in table if not isinstance(r, str)]
    best = [min((round(r[1][j], 1) for r in data if r[1][j] is not None), default=None) for j in range(len(cols))]
    lines = []
    for r in table:
        if isinstance(r, str):
            lines.append(r)
            continue
        cells = []
        for j, v in enumerate(r[1]):
            if v is None:
                cells.append("--")
            else:
                t = f"{v:+.0f}"
                cells.append(f"$\\mathbf{{{t}}}$" if round(v, 1) == best[j] else f"${t}$")
        lines.append(r[0] + " & " + " & ".join(cells) + " \\\\")
    out = OUT / "cross_model_rows.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prov = {"inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(used)},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "columns": [f"{m} P={sp}" for m, _, _, sp in cols]}
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
