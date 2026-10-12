"""Paired zero-shot task accuracy for original and decoded V100 checkpoints.

Use stock lm-evaluation-harness tasks on the same 256 examples per task,
chosen once with seed 42. Preserve sample identities, likelihoods and accuracy;
skip completed method/task cases with matching input and evaluation identities.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import random
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def semantic_identity(identity):
    value = json.loads(json.dumps(identity))
    value["protocol"].pop("script_sha256", None)
    return re.sub(r"<function ([^ ]+) at 0x[0-9a-fA-F]+>", r"<function \1>", json.dumps(value, sort_keys=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--encoded-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--methods", default="no_compression,iterative_ucb_disjoint,excp_style")
    ap.add_argument("--tasks", default="hellaswag,arc_easy,piqa")
    ap.add_argument("--examples", type=int, default=256)
    ap.add_argument("--task-data", type=Path)
    ap.add_argument("--subset", choices=("main", "calibration", "final"), default="main")
    ap.add_argument("--residual-scale", type=float, default=1.)
    args = ap.parse_args()
    import torch
    import lm_eval
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable
    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    from experiments.lib.residual_runtime import sha256_file, write_json
    from run_system_bytes_comparison import excp_dequant

    methods, tasks = args.methods.split(","), args.tasks.split(",")
    if set(methods) - {"no_compression", "iterative_ucb_disjoint", "excp_style"} or len(set(methods)) != len(methods) or args.examples < 1 or not 0 < args.residual_scale <= 2:
        ap.error("invalid methods or example count")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manager = TaskManager()
    if args.task_data is None:
        task_dict = get_task_dict(tasks, manager)
    else:
        from lm_eval.tasks._yaml_loader import load_yaml
        from lm_eval.api.task import ConfigurableTask
        local_data = json.loads(args.task_data.read_text())
        definitions = {"hellaswag": "hellaswag/hellaswag.yaml", "arc_easy": "arc/arc_easy.yaml", "piqa": "piqa/piqa.yaml"}
        task_dict = {}
        for task in tasks:
            entry = local_data[task]
            data_path = args.task_data.parent / entry["file"]
            if sha256_file(data_path) != entry["sha256"]:
                raise RuntimeError("local benchmark file hash differs")
            cfg = load_yaml(Path(lm_eval.__file__).parent / "tasks" / definitions[task])
            cfg.pop("tag", None)
            cfg.update(dataset_path="parquet", dataset_name=None, training_split=None,
                       dataset_kwargs={"data_files": {entry["split"]: str(data_path)}})
            if entry["split"] == "test":
                cfg["validation_split"] = None
            task_dict[task] = ConfigurableTask(config=cfg)
    subsets = {}
    for task in tasks:
        obj = task_dict[task]
        count = len(obj.eval_docs)
        if args.subset == "final":
            main_indices = set(random.Random(42).sample(range(count), min(256, count)))
            calibration_pool = [i for i in range(count) if i not in main_indices]
            excluded = main_indices | set(random.Random(42).sample(calibration_pool, min(64, len(calibration_pool))))
            excluded |= set(random.Random(42).sample(calibration_pool, min(128, len(calibration_pool))))
            candidates = [i for i in range(count) if i not in excluded]
            indices = sorted(random.Random(42).sample(candidates, min(args.examples, len(candidates))))
            assert not excluded.intersection(indices)
        elif args.subset == "calibration":
            main_indices = set(random.Random(42).sample(range(count), min(256, count)))
            candidates = [i for i in range(count) if i not in main_indices]
            indices = sorted(random.Random(42).sample(candidates, min(args.examples, len(candidates))))
            assert not main_indices.intersection(indices)
        else:
            indices = sorted(random.Random(42).sample(range(count), min(args.examples, count)))
        docs = [(i, obj.eval_docs[i]) for i in indices]
        digest = hashlib.sha256(json.dumps(docs, sort_keys=True, default=handle_non_serializable).encode()).hexdigest()
        subsets[task] = {"indices": indices, "documents_sha256": digest, "split_size": count,
                         "config": obj.config.to_dict()}
    protocol = {"seed": 42, "num_fewshot": 0, "batch_size": 8, "dtype": "float32",
                "examples_requested": args.examples, "harness_version": importlib.metadata.version("lm_eval"),
                "tasks": subsets, "script_sha256": sha256_file(Path(__file__)),
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "base_files": {p.name: sha256_file(p) for p in sorted(args.model_root.glob("*")) if p.is_file()}}
    protocol = json.loads(json.dumps(protocol, default=handle_non_serializable))
    if args.subset == "calibration":
        protocol["subset"] = "calibration; disjoint from the existing 256-example main subset"
    elif args.subset == "final":
        protocol["subset"] = "final; excludes the 256-example development set and all 64/128-example calibration indices"
    if args.residual_scale != 1.:
        protocol["residual_scale"] = args.residual_scale
    if args.task_data is not None:
        protocol["local_dataset_provenance"] = local_data
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    selected = json.loads((args.encoded_dir / "completed.json").read_text())["selected"] if any(m != "no_compression" for m in methods) else {}
    todo = []
    for method in methods:
        archive = None
        if method != "no_compression":
            entry = selected[method]
            record_path = args.encoded_dir / entry["file"]
            if sha256_file(record_path) != entry["sha256"]:
                raise RuntimeError("selected archive record changed")
            rec = json.loads(record_path.read_text())
            archive = args.encoded_dir / rec["archive"]["file"]
            if sha256_file(archive) != rec["archive"]["sha256"] or rec["identity"]["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
                raise RuntimeError("selected archive identity differs")
        identity = {"protocol": protocol, "method": method,
                    "archive_sha256": sha256_file(archive) if archive is not None else None}
        missing = []
        for task in tasks:
            path = args.out_dir / f"{method}_{task}.json"
            if path.exists():
                old = json.loads(path.read_text())
                frozen_source = subprocess.check_output(["git", "show", old["code_commit"] + ":experiments/scripts/run_v100_downstream_subset.py"], cwd=ROOT)
                if hashlib.sha256(frozen_source).hexdigest() != old["identity"]["protocol"]["script_sha256"]:
                    raise RuntimeError("frozen evaluation source hash differs")
                if semantic_identity(old["identity"]) != semantic_identity(identity) or old["status"] != "completed":
                    raise RuntimeError("downstream result identity conflict")
                print("SKIP " + str(path), flush=True)
            else:
                missing.append(task)
        if missing:
            todo.append((method, archive, identity, missing))
    if not todo:
        return
    write_json(args.out_dir / "run-manifest.json", {"protocol": protocol, "code_commit": commit,
                                                   "gpu": torch.cuda.get_device_name(0)})
    model = GPTNeoXForCausalLM.from_pretrained(args.model_root, dtype=torch.float32, local_files_only=True)
    base = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.to("cuda:0").eval()
    for method, archive, identity, missing in todo:
        if archive is None:
            original = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(original["model_state_dict"], strict=True)
            del original
        else:
            from experiments.lib.calibrated_archive import load_payload
            payload = load_payload(archive)
            params = dict(model.named_parameters())
            with torch.no_grad():
                for n, value in payload["model"].items():
                    gain = args.residual_scale * payload.get("residual_scale", 1.)
                    params[n].copy_(base[n].to("cuda:0") + gain * excp_dequant(value, base[n].shape, "cuda:0").float())
            model.load_state_dict({**model.state_dict(), **payload["buffers"]}, strict=True)
            del payload
        lm = HFLM(pretrained=model, tokenizer=tokenizer, device="cuda:0", batch_size=8)
        for task in missing:
            began = time.perf_counter()
            result = lm_eval.simple_evaluate(model=lm, tasks=[task_dict[task]], task_manager=manager, num_fewshot=0,
                                             samples={task: subsets[task]["indices"]}, log_samples=True,
                                             random_seed=42, numpy_random_seed=42, torch_random_seed=42,
                                             fewshot_random_seed=42, bootstrap_iters=0)
            samples = [{k: x[k] for k in ("doc_id", "doc_hash", "target", "resps", "filtered_resps", "acc", "acc_norm") if k in x}
                       for x in result["samples"][task]]
            if len(samples) != len(subsets[task]["indices"]) or sorted(x["doc_id"] for x in samples) != subsets[task]["indices"]:
                raise RuntimeError("evaluated task sample identities differ")
            record = {"status": "completed", "identity": identity, "code_commit": commit,
                      "gpu": torch.cuda.get_device_name(0), "elapsed_s": time.perf_counter() - began,
                      "metrics": result["results"][task], "n_samples": result["n-samples"][task],
                      "harness_task_version": result["versions"][task], "samples": samples}
            record = json.loads(json.dumps(record, default=handle_non_serializable))
            path = args.out_dir / f"{method}_{task}.json"
            write_json(path, record)
            print("RESULT " + str(path) + " " + json.dumps(record["metrics"]), flush=True)


if __name__ == "__main__":
    main()
