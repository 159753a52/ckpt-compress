"""Diagnose the signed-benefit coordinate fraction at a recovery point.

Reports the fraction of eligible residual coordinates whose signed Taylor
contribution is negative (i.e. predicted to reduce loss when reverted) under
the exact paper protocol segments, so the rollback-mechanism story can be
checked per checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.data import cache_batches, get_data_loaders  # noqa: E402
from experiments.lib.models import load_model  # noqa: E402
from experiments.lib.residual_runtime import (  # noqa: E402
    configure_hf_offline,
    load_training_checkpoint,
    set_seed,
)
from experiments.lib.residual_scoring import (  # noqa: E402
    compute_block_taylor_scores,
    eligible_layers,
)
from experiments.lib.residual_training import (  # noqa: E402
    build_optimizer,
    clone_model_state_to_cpu,
    partition_repeated_seed_batches,
    train_segment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--model-family", default=None,
                        help="Defaults to the --model value (registry keys are family names).")
    parser.add_argument("--dataset", default="wikitext103")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/gpt2_medium_wikitext103_1000steps/checkpoint_step_1000.pt",
    )
    parser.add_argument("--data-dir", default="data/wikitext103")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--recovery-count", type=int, default=3)
    parser.add_argument("--hvp-batches", type=int, default=8)
    parser.add_argument("--train-pool", type=int, default=124)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_offline()
    device = args.device
    model_family = args.model_family or args.model

    context_checkpoint = load_training_checkpoint(Path(args.checkpoint))
    model = load_model(
        args.model,
        pretrained=False,
        checkpoint_path=None,
        device="cpu",
        dataset_name=args.dataset,
    )[0]
    model.load_state_dict(context_checkpoint.model_state, strict=True)
    model.to(device)

    set_seed(0)
    train_loader, _val_loader, task_type = get_data_loaders(
        args.model, args.dataset, batch_size=2, seq_length=128, data_dir=args.data_dir
    )
    pool = cache_batches(train_loader, args.train_pool, task_type)

    plan = partition_repeated_seed_batches(
        pool, args.total_steps, args.recovery_count, args.hvp_batches, args.seed
    )
    optimizer = build_optimizer(model, context_checkpoint.optimizer_state, args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=0.0
    )

    # The reconstruction reference is the pre-recovery checkpoint state; the
    # residual is the update accumulated by the first training segment.
    reconstructed = {
        name: values.detach().float().clone()
        for name, values in context_checkpoint.model_state.items()
    }

    training = train_segment(
        model,
        optimizer,
        scheduler,
        plan.training_segments[0],
        args.seed,
        device,
        task_type=task_type,
    )

    layers = eligible_layers(model, model_family)
    current = clone_model_state_to_cpu(model)
    delta = {
        name: current[name].detach().float() - reconstructed[name].detach().float()
        for layer in layers
        for name in layer
    }
    scores, scoring_meta = compute_block_taylor_scores(
        model,
        plan.scoring_batches[0],
        layers,
        delta,
        device,
        aggregation="signed_mean",
        task_type=task_type,
        model_family=model_family,
    )

    per_layer = []
    total_negative = 0
    total_elements = 0
    for index, names in enumerate(layers):
        flat = torch.cat([scores[name].flatten() for name in names])
        negative = int((flat < 0).sum().item())
        total_negative += negative
        total_elements += flat.numel()
        per_layer.append(
            {
                "layer": index,
                "elements": flat.numel(),
                "negative": negative,
                "negative_fraction": negative / flat.numel(),
                "mean": float(flat.mean()),
                "min": float(flat.min()),
            }
        )

    payload = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "recovery_cycle": 1,
        "segment_steps": len(plan.training_segments[0]),
        "scoring_batches": len(plan.scoring_batches[0]),
        "pre_recovery_loss": training.get("losses", [None])[-1] if training else None,
        "overall_negative_fraction": total_negative / total_elements,
        "eligible_elements": total_elements,
        "per_layer": per_layer,
        "scoring_metadata": {
            key: value
            for key, value in scoring_meta.items()
            if key not in {"layer_seconds"}
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1))
    print(
        f"{args.model}: negative-benefit fraction "
        f"{payload['overall_negative_fraction']:.4f} over {total_elements} coords"
    )


if __name__ == "__main__":
    main()
