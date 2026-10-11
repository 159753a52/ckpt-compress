"""Regenerate the V100 quality/bytes and paired task-accuracy tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

METHODS = [("no_compression", "Original"), ("excp_style", "ExCP-style"),
           ("iterative_ucb_disjoint", "DACP (staged)")]
TASKS = [("hellaswag", "acc_norm,none"), ("arc_easy", "acc_norm,none"), ("piqa", "acc,none")]


def generate(encoded, downstream, out):
    inputs = {}

    def load(path):
        raw = path.read_bytes()
        inputs[path.as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    completed = load(encoded / "completed.json")
    assert completed["status"] == "completed"
    assert completed["identity"]["moment_storage"] == "all Adam entries retained in bfloat16"
    records = {}
    for item in completed["records"]:
        path = encoded / item["file"]
        row = load(path)
        assert inputs[path.as_posix()] == item["sha256"] and row["identity"] == completed["identity"]
        assert row["status"] == "completed"
        reused = row["weight_quality_reused"]
        relative = reused["file"].replace("\\", "/").split("/results/", 1)[-1]
        source_path = Path(__file__).resolve().parents[3] / "results" / relative
        source = load(source_path)
        assert inputs[source_path.as_posix()] == reused["sha256"]
        assert row["validation"] == source["validation"] and row["gate"] == source["gate"]
        assert row["weight_payload_reused_from_archive_sha256"] == source["archive"]["sha256"]
        assert row["quantization"]["moment_bits"] == 16 and row["quantization"]["joint_moment_beta"] is None
        assert math.isclose(row["archive"]["byte_compression"], row["archive"]["original_checkpoint_bytes"] / row["archive"]["bytes"])
        records[item["file"]] = row
    chosen = {method: records[item["file"]] for method, item in completed["selected"].items()}
    ours = chosen["iterative_ucb_disjoint"]
    original_bytes = ours["archive"]["original_checkpoint_bytes"]
    original_ppl = ours["baseline_validation_reused"]["ppl"]
    assert ours["validation_relative_ppl_change"] < .01
    lines = [r"\begin{table*}[!tp]", r"\centering\small",
             r"\caption{Single-V100 Pythia-410M/Alpaca archive experiment, seed 42. Both selectors use 70\% eligible residual sparsity and the same weight-codec grid, selected using a separate 1\% calibration-perplexity gate. Size includes all parameters and buffers, both Adam moments, steps and parameter groups; the shared pretrained reference is external. Every Adam entry is retained in BF16. $w/m$ gives weight-code bits and moment precision; one GiB is $2^{30}$ bytes. ExCP-style transfers residual-magnitude selection into this shared codec.}",
             r"\label{tab:v100-encoded-quality}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Selection & $w/m$ & Size (GiB) & Byte compression & PPL & $\Delta$ PPL (\%) \\", r"\midrule",
             f"Original & 32/32 & {original_bytes / 2**30:.3f} & $1.00\\times$ & {original_ppl:.4f} & 0.000" + r" \\"]
    for method, label in METHODS[1:]:
        row = chosen[method]
        assert row["passed_gate"] and row["eligible_kept"] == ours["eligible_kept"]
        ppl = row["validation"]["perplexity"]
        assert math.isclose(math.expm1(row["validation"]["loss"] - row["baseline_validation_reused"]["val"]), row["validation_relative_ppl_change"])
        lines.append(f"{label} & {row['quantization']['weight_bits']}/{row['quantization']['moment_bits']} & "
                     f"{row['archive']['bytes'] / 2**30:.3f} & ${row['archive']['byte_compression']:.2f}\\times$ & "
                     f"{ppl:.4f} & {100 * row['validation_relative_ppl_change']:+.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "v100_encoded_quality_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    accuracy = {}
    if downstream is not None:
        lines = [r"\begin{table}[!tp]", r"\centering\small",
                 r"\caption{Zero-shot downstream accuracy (\%) of the original and physically decoded archives in Table~\ref{tab:v100-encoded-quality}. Each task uses the same 256 examples sampled once with seed 42, using stock lm-evaluation-harness 0.4.13 prompts. HellaSwag and ARC-Easy use length-normalized accuracy; PIQA uses accuracy. These are subset results. $\Delta$ Avg. is in percentage points.}",
                 r"\label{tab:v100-downstream}", r"\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrrr}", r"\toprule",
                 r"Selection & HellaSwag & ARC-Easy & PIQA & Avg. & $\Delta$ Avg. \\", r"\midrule"]
        reference = {}
        for method, label in METHODS:
            values = []
            for task, metric in TASKS:
                row = load(downstream / f"{method}_{task}.json")
                assert row["status"] == "completed" and len(row["samples"]) == 256
                docs = [(x["doc_id"], x["doc_hash"]) for x in row["samples"]]
                if method == "no_compression":
                    reference[task] = (row["identity"]["protocol"], docs)
                else:
                    assert reference[task] == (row["identity"]["protocol"], docs)
                    assert row["identity"]["archive_sha256"] in {chosen[method]["archive"]["sha256"], chosen[method].get("weight_payload_reused_from_archive_sha256")}
                sample_metric = metric.split(",")[0]
                value = row["metrics"][metric]
                assert math.isclose(value, sum(x[sample_metric] for x in row["samples"]) / 256)
                values.append(100 * value)
            avg = sum(values) / 3
            accuracy[method] = {"values_percent": values, "average_percent": avg}
            delta = avg - accuracy["no_compression"]["average_percent"]
            lines.append(label + " & " + " & ".join(f"{v:.2f}" for v in values + [avg, delta]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        (out / "v100_downstream_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("v100_*quality_table.tex")}
    if downstream is not None:
        p = out / "v100_downstream_table.tex"
        outputs[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"inputs": inputs, "outputs": outputs, "accuracy": accuracy,
              "selected_archive_records": completed["selected"], "scope": completed["identity"]["state_scope"]}
    (out / "v100_quality.provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoded", type=Path, required=True)
    ap.add_argument("--downstream", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(generate(args.encoded, args.downstream, args.out_dir)["accuracy"]))


if __name__ == "__main__":
    main()
