"""Audit manuscript inputs against existing records, without model or GPU work."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO.parents[1]
FIGS = PAPER / "paper" / "figs" / "gen"
PROVENANCE = ["cross_model_rows", "cross_model_rows_extra", "tolerance_rows",
              "tolerance_rows_extra", "f1_base_chains_32b", "v100_fixed_budget",
              "e2_pythia_ablation", "f2_small_chains", "staged_system_curve",
              "a2_taylor_fidelity_32b", "b3_scoring_cost_32b"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_path(recorded):
    normalized = recorded.replace("\\", "/")
    relative = normalized.split("/results/", 1)[-1]
    if relative.startswith("results/"):
        relative = relative[len("results/"):]
    return REPO / "results" / relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    failures, sources, outputs, tables = [], {}, {}, []
    for name in PROVENANCE:
        provenance = json.loads((FIGS / f"{name}.provenance.json").read_text(encoding="utf-8"))
        inputs = list(provenance.get("inputs", {}).items())
        for source in provenance.get("sources", []):
            recorded = source.get("path") or provenance["results_directory"] + "/" + source["file"]
            inputs.append((recorded, source["sha256"]))
        for recorded, expected in inputs:
            path = result_path(recorded)
            actual = sha256(path) if path.is_file() else None
            sources[path.relative_to(REPO).as_posix()] = actual
            if actual != expected:
                failures.append({"provenance": name, "input": recorded, "expected": expected, "actual": actual})
        for filename, expected in provenance.get("outputs", {}).items():
            path = FIGS / filename
            actual = sha256(path) if path.is_file() else None
            outputs[filename] = actual
            if actual != expected:
                failures.append({"output": filename, "expected": expected, "actual": actual})

    import cross_model_table
    import tolerance_table
    import v100_confirmation_table
    import base_chains_fig

    original_argv = sys.argv[:]
    with tempfile.TemporaryDirectory(prefix="dacp-evidence-") as directory:
        generated = Path(directory)
        with contextlib.redirect_stdout(io.StringIO()):
            for module in (cross_model_table, tolerance_table):
                module.OUT = generated
                for extra in (False, True):
                    module.used.clear()
                    sys.argv = [module.__file__] + (["--extra"] if extra else [])
                    module.main()
            sys.argv = [v100_confirmation_table.__file__, "--results",
                        str(REPO / "results" / "v100_staged_confirmation_20261010"),
                        "--out-dir", str(generated)]
            v100_confirmation_table.main()
        nc, records, _ = base_chains_fig.load(REPO / "results" / "probe_coverage_32b" / "run1" / "base_chains")
        (generated / "base_chains_rows.tex").write_text(base_chains_fig.tex_rows(nc, records, [0.95, 0.98]), encoding="utf-8")
        manuscript = (PAPER / "paper" / "EMNLP_26_front.tex").read_text(encoding="utf-8")
        for path in sorted(generated.glob("*.tex")):
            matches = path.read_text(encoding="utf-8") == (FIGS / path.name).read_text(encoding="utf-8")
            included = (manuscript.count("\\input{figs/gen/" + path.stem + "}")
                        + manuscript.count("\\inputtablerows{figs/gen/" + path.name + "}")) == 1
            tables.append({"file": path.name, "matches_raw_regeneration": matches, "included_once": included})
            if not matches or not included:
                failures.append({"table": path.name, "matches": matches, "included_once": included})
    sys.argv = original_argv
    report = {"passed": not failures, "scope": "existing records only; no model loading or GPU experiments",
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "audit_script_sha256": sha256(Path(__file__)),
              "provenance_files": len(PROVENANCE), "unique_result_files": len(sources),
              "tables": tables, "source_sha256": sources, "output_sha256": outputs, "failures": failures}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "provenance_files", "unique_result_files", "tables", "failures")}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
