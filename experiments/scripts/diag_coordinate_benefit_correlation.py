"""Coordinate-level benefit-score validity diagnostic.

At one paper-protocol recovery point, sample eligible residual coordinates,
measure the ACTUAL loss change of reverting each one, and rank-correlate that
ground truth with three selection signals:

- ``signed``: signed Taylor contribution (the optimized DACP signal)
- ``abs_taylor``: |signed contribution| (the original paper signal)
- ``gw``: full-weight gradient-weight product |g*theta| (Inshrinkerator-style)

A positive correlation with ``revert_delta_loss`` means higher-scored
coordinates hurt more when reverted (a valid damage ranking); for ``signed``
the meaningful direction is negative correlation with raw revert benefit.
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
from experiments.lib.residual_runtime import task_loss  # noqa: E402
from experiments.lib.models import load_model  # noqa: E402
from experiments.lib.residual_runtime import (  # noqa: E402
    configure_hf_offline,
    load_training_checkpoint,
    set_seed,
)
from experiments.lib.residual_scoring import (  # noqa: E402
    compute_block_first_order_scores,
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
    parser.add_argument("--model-family", default=None)
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
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--probe-batches", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_offline()
    device = args.device
    model_family = args.model_family
    if model_family is None:
        family_map = {
            "gpt2-medium": "gpt2",
            "gpt2-large": "gpt2",
            "pythia-410m": "pythia",
            "pythia-1b": "pythia",
            "bert-large": "bert",
        }
        model_family = family_map.get(args.model, args.model)

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
    train_loader, val_loader, task_type = get_data_loaders(
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
    reconstructed = {
        name: values.detach().float().clone()
        for name, values in context_checkpoint.model_state.items()
    }
    train_segment(
        model,
        optimizer,
        scheduler,
        plan.training_segments[0],
        args.seed,
        device,
        task_type=task_type,
    )

    layers = eligible_layers(model, model_family)
    names = [name for layer in layers for name in layer]
    named_params = dict(model.named_parameters())
    current = clone_model_state_to_cpu(model)
    delta = {
        name: (current[name].detach().float() - reconstructed[name].detach().float()).to(device)
        for name in names
    }
    # Free large intermediate copies before the sampling loop.
    del current, reconstructed, context_checkpoint, optimizer, scheduler
    import gc as _gc

    _gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    scores_signed, _meta = compute_block_taylor_scores(
        model,
        plan.scoring_batches[0],
        layers,
        {name: value.cpu() for name, value in delta.items()},
        device,
        aggregation="signed_mean",
        task_type=task_type,
        model_family=model_family,
    )
    scores_gw, _ = compute_block_first_order_scores(
        model,
        plan.scoring_batches[0],
        layers,
        {name: named_params[name].detach().cpu() for name in names},
        device,
        task_type=task_type,
        model_family=model_family,
    )

    generator = torch.Generator().manual_seed(args.seed)
    samples = []
    for name in names:
        numel = delta[name].numel()
        count = max(1, int(args.samples * numel / sum(delta[n].numel() for n in names)))
        indices = torch.randperm(numel, generator=generator)[:count]
        samples.extend((name, int(index)) for index in indices)

    probe_batches = []
    for offset in range(args.probe_batches):
        index = (args.seed * 7 + offset) % len(pool)
        probe_batches.append(pool[index])

    model.eval()

    def probe_loss() -> float:
        total = 0.0
        with torch.no_grad():
            for batch in probe_batches:
                total += float(task_loss(model, batch, task_type, device))
        return total / len(probe_batches)

    baseline = probe_loss()
    records = []
    with torch.no_grad():
        for name, flat_index in samples:
            parameter = named_params[name]
            view = parameter.view(-1)
            shift = -delta[name].view(-1)[flat_index]
            if float(shift.abs()) == 0.0:
                continue
            original = view[flat_index].item()
            view[flat_index] = original + float(shift)
            reverted = probe_loss()
            view[flat_index] = original
            records.append(
                {
                    "name": name,
                    "index": flat_index,
                    "signed": float(scores_signed[name].view(-1)[flat_index]),
                    "abs_taylor": float(scores_signed[name].view(-1)[flat_index].abs()),
                    "gw": float(scores_gw[name].view(-1)[flat_index]),
                    "revert_delta_loss": reverted - baseline,
                }
            )

    def spearman(xs, ys):
        xs_rank = torch.tensor(xs).argsort().argsort().float()
        ys_rank = torch.tensor(ys).argsort().argsort().float()
        xs_centered = xs_rank - xs_rank.mean()
        ys_centered = ys_rank - ys_rank.mean()
        denom = (xs_centered.norm() * ys_centered.norm()).item()
        if denom == 0.0:
            return float("nan")
        return float((xs_centered * ys_centered).sum() / denom)

    payload = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "samples": len(records),
        "baseline_loss": baseline,
        "correlations": {
            key: {
                "spearman_vs_revert_delta_loss": spearman(
                    [r[key] for r in records], [r["revert_delta_loss"] for r in records]
                ),
            }
            for key in ("signed", "abs_taylor", "gw")
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1))
    for key, value in payload["correlations"].items():
        print(f"{args.model} {key}: spearman {value['spearman_vs_revert_delta_loss']:+.4f}")
    del val_loader


if __name__ == "__main__":
    main()
