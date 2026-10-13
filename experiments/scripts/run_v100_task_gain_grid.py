"""Select an equal-budget decoder-gain grid on the existing task calibration set."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--source-record", type=Path, required=True)
    ap.add_argument("--matrix-dir", type=Path, required=True)
    ap.add_argument("--source-calibration", type=Path, required=True)
    ap.add_argument("--task-data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    import lm_eval
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable
    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    from experiments.lib.calibrated_archive import load_payload
    from experiments.lib.residual_runtime import batch_hash, evaluate_lm, sha256_file, write_json
    from experiments.lib.residual_training import partition_repeated_seed_batches
    from run_residual_sweep_small import REGIMES, load_regime
    from run_system_bytes_comparison import excp_dequant

    source = json.loads(args.source_record.read_text())
    source_cal = json.loads(args.source_calibration.read_text())
    matrix = json.loads((args.matrix_dir / "archive.json").read_text())
    selected = json.loads((args.matrix_dir / "matrix_gains.manifest.json").read_text())
    assert sha256_file(args.matrix_dir / "matrix_gains.manifest.json") == matrix["archive"]["sha256"]
    assert matrix["matrix_gain_calibration"]["identity"]["source_record_sha256"] == sha256_file(args.source_record)
    assert source_cal["method"] == source["method"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "completed.json").exists():
        raise RuntimeError("completed grid exists; retain its recorded selection")
    REGIMES["R1"]["checkpoint"] = str(args.checkpoint.resolve())
    ctx = load_regime("R1", args.checkpoint.parent, args.data_root, "cuda:0")
    s, pool = ctx["sweep"], ctx["pool"]
    assert sha256_file(args.checkpoint) == source["identity"]["checkpoint_sha256"]
    plan = partition_repeated_seed_batches(pool, 100, 1, 16, 42)
    offset = len(plan.training_segments[0])
    used = set(plan.selected_pool_indices[offset:offset + 16])
    gate = [pool[i] for i in range(len(pool)) if i not in used][:64]
    assert batch_hash(gate) == source_cal["identity"]["gate_sha256"]
    baseline_gate = evaluate_lm(s.model, gate, s.device)
    original_manifest = json.loads((args.source_record.parent / source["archive"]["file"]).read_text())
    initial = original_manifest["gain"]
    assert initial == source_cal["selected"]["gain"]
    raw_path = args.matrix_dir / "unscaled.manifest.json"
    raw_manifest = json.loads(raw_path.read_text())
    payload = load_payload(raw_path)
    base_model = GPTNeoXForCausalLM.from_pretrained(args.data_root / "models/pythia-410m", dtype=torch.float32, local_files_only=True)
    base = {n: p.detach().to(s.device) for n, p in base_model.named_parameters()}
    del base_model
    params = dict(s.model.named_parameters())
    manager = TaskManager()
    local_data = json.loads(args.task_data.read_text())
    tasks = [("hellaswag", "hellaswag/hellaswag.yaml", "acc_norm"), ("arc_easy", "arc/arc_easy.yaml", "acc_norm"), ("piqa", "piqa/piqa.yaml", "acc")]
    task_dict, subsets = {}, {}
    for task, definition, metric in tasks:
        entry = local_data[task]
        file = args.task_data.parent / entry["file"]
        assert sha256_file(file) == entry["sha256"]
        cfg = load_yaml(Path(lm_eval.__file__).parent / "tasks" / definition)
        cfg.pop("tag", None)
        cfg.update(dataset_path="parquet", dataset_name=None, training_split=None,
                   dataset_kwargs={"data_files": {entry["split"]: str(file)}})
        if entry["split"] == "test":
            cfg["validation_split"] = None
        obj = task_dict[task] = ConfigurableTask(config=cfg)
        count = len(obj.eval_docs)
        excluded = set(random.Random(42).sample(range(count), min(256, count)))
        candidates = [i for i in range(count) if i not in excluded]
        indices = sorted(random.Random(42).sample(candidates, 128))
        docs = [(i, obj.eval_docs[i]) for i in indices]
        subsets[task] = {"indices": indices, "documents_sha256": hashlib.sha256(json.dumps(docs, sort_keys=True, default=handle_non_serializable).encode()).hexdigest()}
    tokenizer = AutoTokenizer.from_pretrained(args.data_root / "models/pythia-410m", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    lm = HFLM(pretrained=s.model, tokenizer=tokenizer, device=s.device, batch_size=8)
    cases = [(a, b) for a in (0., .25, .5, .75, 1.) for b in (-.05, 0., .05)]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identity = {"checkpoint_sha256": sha256_file(args.checkpoint), "source_record_sha256": sha256_file(args.source_record),
                "matrix_record_sha256": sha256_file(args.matrix_dir / "archive.json"), "script_sha256": sha256_file(Path(__file__)),
                "gate_sha256": batch_hash(gate), "seed": 42, "cases": cases, "task_subsets": subsets,
                "harness_version": importlib.metadata.version("lm_eval"), "ppl_gate_percent": .5}
    write_json(args.out_dir / "run-manifest.json", {"identity": identity, "code_commit": commit, "method": source["method"],
               "selection": "highest calibration task mean, then lower gate loss, smaller absolute offset, smaller interpolation"})
    points = []
    for a, b in cases:
        tag = f"a{a:g}_b{b:g}"
        folder = args.out_dir / tag
        folder.mkdir(exist_ok=True)
        gains = {n: max(.5, min(1.5, initial + a * (g - initial) + b)) for n, g in selected["parameter_gains"].items()}
        manifest = {**raw_manifest, "parameter_gains": gains}
        size = 0
        for key in ("state", "weights"):
            item = manifest[key]
            original = args.matrix_dir / item["file"]
            dest = folder / item["file"]
            if not dest.exists():
                os.link(original, dest)
            assert os.path.samefile(original, dest)
            size += dest.stat().st_size
        path = folder / "task_gains.manifest.json"
        write_json(path, manifest)
        size += path.stat().st_size
        with torch.no_grad():
            for n, p in params.items():
                encoded = dict(payload["model"][n])
                if n in gains:
                    key = "raw" if "raw" in encoded else "codebook"
                    encoded[key] = encoded[key].float() * gains[n]
                p.copy_(base[n] + excp_dequant(encoded, p.shape, s.device).float())
        s.model.load_state_dict({**s.model.state_dict(), **payload["buffers"]}, strict=True)
        quality = evaluate_lm(s.model, gate, s.device)
        change = 100 * math.expm1(quality["loss"] - baseline_gate["loss"])
        point = {"case": tag, "a": a, "b": b, "gate": quality, "gate_ppl_change_percent": change, "passed_gate": change <= .5,
                 "archive": {"file": path.name, "bytes": size, "sha256": sha256_file(path)}}
        if point["passed_gate"]:
            values = []
            for task, definition, metric in tasks:
                result = lm_eval.simple_evaluate(model=lm, tasks=[task_dict[task]], task_manager=manager, num_fewshot=0,
                    samples={task: subsets[task]["indices"]}, log_samples=True, random_seed=42, numpy_random_seed=42,
                    torch_random_seed=42, fewshot_random_seed=42, bootstrap_iters=0)
                samples = [{k: x[k] for k in ("doc_id", "doc_hash", "target", "resps", "filtered_resps", "acc", "acc_norm") if k in x}
                           for x in result["samples"][task]]
                assert sorted(x["doc_id"] for x in samples) == subsets[task]["indices"]
                value = sum(x[metric] for x in samples) / 128
                assert math.isclose(value, result["results"][task][metric + ",none"])
                values.append(value)
                write_json(folder / (task + ".json"), {"identity": identity, "case": tag, "task": task,
                    "metrics": result["results"][task], "samples": samples, "code_commit": commit})
            point.update(task_accuracy=values, task_mean=sum(values) / 3)
        if a == 0 and b == 0:
            assert abs(quality["loss"] - source["gate"]["loss"]) <= 1e-6
            assert point["task_accuracy"] == source_cal["selected"]["task_accuracy"]
        points.append(point)
        write_json(args.out_dir / "points.json", {"identity": identity, "points": points})
        print("GRID", source["method"], tag, change, point.get("task_mean"), flush=True)
    best = max((p for p in points if p["passed_gate"]), key=lambda p: (p["task_mean"], -p["gate"]["loss"], -abs(p["b"]), -p["a"]))
    folder = args.out_dir / best["case"]
    actual = load_payload(folder / best["archive"]["file"])
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(base[n] + excp_dequant(actual["model"][n], p.shape, s.device).float())
    actual_gate = evaluate_lm(s.model, gate, s.device)
    assert abs(actual_gate["loss"] - best["gate"]["loss"]) <= 1e-6
    validation = evaluate_lm(s.model, ctx["eval_batches"], s.device)
    record = {**source, **best, "code_commit": commit, "validation": validation,
              "validation_ppl_change_percent": 100 * math.expm1(validation["loss"] - ctx["uncompressed"]["val"]),
              "byte_compression": args.checkpoint.stat().st_size / best["archive"]["bytes"], "task_grid_identity": identity}
    record.pop("gain", None)
    write_json(folder / "archive.json", record)
    write_json(folder / "completed.json", {"status": "completed", "selected": {source["method"]: {"file": "archive.json", "sha256": sha256_file(folder / "archive.json")}}})
    write_json(args.out_dir / "completed.json", {"status": "completed", "identity": identity, "selected": best,
               "chosen_encoded_dir": str(folder.resolve()), "code_commit": commit})
    print("SELECTED_GRID", source["method"], best["case"], best["task_mean"], record["validation_ppl_change_percent"], flush=True)


if __name__ == "__main__":
    main()
