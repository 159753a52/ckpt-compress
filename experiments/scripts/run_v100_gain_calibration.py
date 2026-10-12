"""Calibrate residual gain on PPL and disjoint task examples using one seed.

Both ExCP and DACP receive the same gain grid. The main task subset is never
used to choose a gain or an archive. Only feasible candidates run task scoring.
"""
from pathlib import Path
import argparse
import io
import json
import math
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--encoded-dir", type=Path, required=True)
    ap.add_argument("--method", choices=("iterative_ucb_disjoint", "excp_style"), required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gains", default="0.8,1,1.2,1.4")
    ap.add_argument("--ppl-gate", type=float, default=1.)
    ap.add_argument("--gain-scope", choices=("all", "eligible"), default="all")
    ap.add_argument("--task-data", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from run_residual_sweep_small import REGIMES, load_regime
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from run_system_bytes_comparison import excp_dequant
    gains = [float(x) for x in args.gains.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = json.loads((args.encoded_dir / "completed.json").read_text())["selected"][args.method]
    rec_path = args.encoded_dir / selected["file"]
    if sha256_file(rec_path) != selected["sha256"]:
        raise RuntimeError("source archive record hash differs")
    rec = json.loads(rec_path.read_text())
    archive = args.encoded_dir / rec["archive"]["file"]
    if sha256_file(archive) != rec["archive"]["sha256"]:
        raise RuntimeError("source archive hash differs")
    identity = {"checkpoint_sha256": sha256_file(args.checkpoint), "source_archive_sha256": sha256_file(archive),
                "script_sha256": sha256_file(Path(__file__)), "gains": gains, "seed": 42,
                "task_examples": 128, "selection": "maximum disjoint calibration task mean under a 1% calibration PPL gate"}
    if args.ppl_gate != 1.:
        identity["ppl_gate_percent"] = args.ppl_gate
        identity["selection"] = "maximum disjoint calibration task mean under the registered PPL gate"
    if rec["identity"]["checkpoint_sha256"] != identity["checkpoint_sha256"]:
        raise RuntimeError("source checkpoint differs")
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    offset = len(plan.training_segments[0])
    used = set(plan.selected_pool_indices[offset:offset + 16])
    gate = [pool[i] for i in range(len(pool)) if i not in used][:64]
    baseline_gate = evaluate_lm(s.model, gate, "cuda:0")
    identity["gate_sha256"] = batch_hash(gate)
    if args.gain_scope == "eligible":
        identity["gain_parameter_names"] = s.names
    from experiments.lib.calibrated_archive import load_payload
    payload = load_payload(archive)
    from transformers import GPTNeoXForCausalLM
    base_model = GPTNeoXForCausalLM.from_pretrained(args.data_root / "models/pythia-410m", dtype=torch.float32, local_files_only=True)
    base = {n: p.detach().cpu() for n, p in base_model.named_parameters()}
    residual = {n: excp_dequant(value, base[n].shape, "cpu").float() for n, value in payload["model"].items()}
    buffers = payload["buffers"]
    del payload
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    points = {}
    for gain in gains:
        path = args.out_dir / f"gate_gain{gain:g}.json"
        if path.exists():
            point = json.loads(path.read_text())
            if point["identity"] != identity:
                raise RuntimeError("existing gain calibration differs")
        else:
            with torch.no_grad():
                for n, p in s.model.named_parameters():
                    multiplier = gain if args.gain_scope == "all" or n in identity["gain_parameter_names"] else 1.
                    p.copy_(base[n].to(s.device) + multiplier * residual[n].to(s.device))
            s.model.load_state_dict({**s.model.state_dict(), **buffers}, strict=True)
            quality = evaluate_lm(s.model, gate, s.device)
            change = 100 * math.expm1(quality["loss"] - baseline_gate["loss"])
            point = {"identity": identity, "gain": gain, "gate": quality,
                     "gate_ppl_change_percent": change, "passed_gate": change <= args.ppl_gate, "code_commit": commit}
            write_json(path, point)
        points[str(gain)] = point
        print(f"GATE {args.method} gain={gain} change={point['gate_ppl_change_percent']:.5f} accepted={point['passed_gate']}", flush=True)
    # Release the scoring model before the stock task runner loads each candidate.
    del ctx, s, base, residual, base_model
    torch.cuda.empty_cache()
    import run_v100_downstream_subset as downstream
    for gain in gains:
        point = points[str(gain)]
        if not point["passed_gate"]:
            continue
        output = args.out_dir / f"tasks_gain{gain:g}"
        sys.argv = [str(Path(downstream.__file__)), "--checkpoint", str(args.checkpoint),
                    "--model-root", str(args.data_root / "models/pythia-410m"), "--encoded-dir", str(args.encoded_dir),
                    "--out-dir", str(output), "--methods", args.method, "--subset", "calibration",
                    "--examples", "128", "--task-data", str(args.task_data), "--residual-scale", str(gain),
                    "--gain-scope", args.gain_scope]
        downstream.main()
        values = []
        for task, metric in [("hellaswag", "acc_norm,none"), ("arc_easy", "acc_norm,none"), ("piqa", "acc,none")]:
            task_path = output / f"{args.method}_{task}.json"
            row = json.loads(task_path.read_text())
            values.append(row["metrics"][metric])
        point["task_accuracy"] = values
        point["task_mean"] = sum(values) / 3
        print(f"TASKS {args.method} gain={gain} mean={point['task_mean']:.6f}", flush=True)
    feasible = [p for p in points.values() if p["passed_gate"]]
    best = max(feasible, key=lambda p: (p["task_mean"], -p["gate"]["loss"])) if feasible else None
    write_json(args.out_dir / "completed.json", {"status": "completed", "identity": identity,
               "source_record": str(rec_path), "source_record_sha256": sha256_file(rec_path), "method": args.method,
               "points": points, "selected": best, "code_commit": commit,
               "scope": "calibration only; no main-subset or independent validation result used"})


if __name__ == "__main__":
    main()
