"""Evaluate a workload's step checkpoint (before any recovery training) on the
exact cached evaluation batches used by run_paper_experiments.py.

Usage (from the repository root):
  python eval_checkpoint_reference.py --manifest M --workload W \
      --checkpoint-root C --data-root D --output out.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

import run_paper_experiments as rpe  # noqa: E402
from experiments.lib.models import load_model  # noqa: E402
from experiments.lib.residual_runtime import evaluate_task  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rpe.configure_hf_offline()
    manifest = rpe.load_paper_manifest(args.manifest)
    workload = next(w for w in manifest.workloads if w.name == args.workload)
    context = rpe._load_workload_context(
        workload,
        checkpoint_root=args.checkpoint_root,
        data_root=args.data_root,
        declared_checkpoint_root=rpe.ROOT / "checkpoints",
        declared_data_root=rpe.ROOT / "data",
        expected_data_identity=None,
    )
    model = load_model(workload.model, pretrained=False, device="cpu", dataset_name=workload.dataset)[0]
    model.load_state_dict(context.checkpoint.model_state, strict=True)
    model.to(args.device)
    metrics = evaluate_task(model, context.eval_batches, workload.task_type, args.device)
    result = {
        "workload": workload.name,
        "checkpoint": str(context.checkpoint_path),
        "data_identity": context.data_identity,
        "checkpoint_metrics_before_recovery": metrics,
    }
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
