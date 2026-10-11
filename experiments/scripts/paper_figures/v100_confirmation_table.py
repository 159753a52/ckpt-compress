"""Generate the fixed-budget V100 table directly from the 18 recorded runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

METHODS = [
    ("one_shot_mean", "One-shot mean", 1, 16),
    ("one_shot_protected", "One-shot protected", 1, 16),
    ("staged_kappa_0", r"Staged, $\kappa=0$", 4, 4),
    ("staged_kappa_0.5", r"Staged, $\kappa=0.5$", 4, 4),
    ("staged_kappa_1", r"Staged, $\kappa=1$", 4, 4),
    ("staged_kappa_2", r"Staged, $\kappa=2$", 4, 4),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, sources, records = {}, [], {}
    for key, label, stages, batches in METHODS:
        rows = []
        for seed in (42, 43, 44):
            path = args.results / f"{key}_p0.98_seed{seed}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            assert value["status"] == "completed"
            assert value["backward_budget"] == value["unique_scoring_batches"] == 16
            assert value["identity"]["stages"] == stages
            assert value["identity"]["batches_per_stage"] == batches
            rows.append(value)
            records[(key, seed)] = value
            sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        losses = [row["delta_loss"] * 1000 for row in rows]
        seconds = [row["score_seconds"] + row["selection_seconds"] for row in rows]
        summary[key] = {"label": label, "stages": stages, "batches": batches,
                        "loss_mean_millinats": statistics.mean(losses), "loss_std_millinats": statistics.stdev(losses),
                        "seconds_mean": statistics.mean(seconds), "seconds_std": statistics.stdev(seconds),
                        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in rows)}
    assert len({value["identity"]["checkpoint_sha256"] for value in records.values()}) == 1
    assert len({value["evaluation_sha256"] for value in records.values()}) == 1
    for seed in (42, 43, 44):
        assert len({records[(key, seed)]["identity"]["scoring_sha256"] for key, *_ in METHODS}) == 1
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Selection & $K\times B$ & $\Delta$ loss & Time (s) \\", r"\midrule"]
    for key, label, stages, batches in METHODS:
        row = summary[key]
        lines.append(f"{label} & ${stages}\\times{batches}$ & "
                     f"${row['loss_mean_millinats']:.1f}\\pm{row['loss_std_millinats']:.1f}$ & "
                     f"${row['seconds_mean']:.2f}\\pm{row['seconds_std']:.2f}$" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}",
              r"\caption{Pythia-410M/Alpaca, 98\% residual sparsity on a V100: mean $\pm$ s.d. over three scoring seeds. All rows use the same 16 scoring batches and 16 backward passes; staged rows partition them into four disjoint groups. Loss changes are in $10^{-3}$ nats. Time includes scoring, partial-state restoration, and selection, excluding final evaluation and setup.}",
              r"\label{tab:v100-fixed-budget}", r"\end{table}"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "v100_fixed_budget_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert all(line.endswith(r"\\") for line in lines if " & " in line)
    document = {"results": summary, "sources": sources, "results_directory": "results/" + args.results.name,
                "relative_damage_reduction": 1 - summary["staged_kappa_1"]["loss_mean_millinats"] / summary["one_shot_mean"]["loss_mean_millinats"]}
    (args.out_dir / "v100_fixed_budget.provenance.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps(document["results"]))


if __name__ == "__main__":
    main()
