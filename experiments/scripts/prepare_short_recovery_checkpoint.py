"""Prepare a short, reproducible recovery snapshot for the V100 gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lib.residual_recovery import (
    ROOT,
    batch_hash,
    checkpoint_optimizer_state,
    checkpoint_state,
    lm_loss,
    load_token_batches,
    optimizer_state_to_cpu,
    set_seed,
    sha256_file,
)
from dacp.models.gpt2 import get_gpt2_medium
from dacp.utils.data_loader import _load_gpt2_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/gpt2_medium_wikitext103_1000steps/checkpoint_step_1000.pt",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/gpt2_medium_wikitext103_short_recovery/checkpoint_step_1020.pt",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT.parent.parent / "data/wikitext103"
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    set_seed(args.seed)
    source_state = checkpoint_state(args.source_checkpoint)
    source_optimizer = checkpoint_optimizer_state(args.source_checkpoint)
    model = get_gpt2_medium(pretrained=False)
    model.load_state_dict(source_state, strict=True)
    model.to(args.device).train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    optimizer.load_state_dict(source_optimizer)
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    del source_optimizer

    tokenizer = _load_gpt2_tokenizer()
    batches = load_token_batches(
        args.data_dir / "train.txt",
        tokenizer,
        args.batch_size,
        args.seq_length,
        args.steps,
    )
    losses = []
    started = time.perf_counter()
    for step, batch in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = lm_loss(model, batch, args.device)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  step {step:02d}/{args.steps:02d}: loss={losses[-1]:.6f}", flush=True)

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    source_step = int(source_payload.get("step", 0)) if isinstance(source_payload, dict) else 0
    del source_payload
    payload = {
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer_state_to_cpu(optimizer.state_dict()),
        "step": source_step + args.steps,
        "loss": losses[-1],
        "short_recovery_provenance": {
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "source_sha256": sha256_file(args.source_checkpoint),
            "train_batches_sha256": batch_hash(batches),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "seconds": time.perf_counter() - started,
            "train_losses": losses,
        },
    }
    torch.save(payload, args.output_checkpoint)
    metadata = {
        **payload["short_recovery_provenance"],
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_sha256": sha256_file(args.output_checkpoint),
        "output_bytes": args.output_checkpoint.stat().st_size,
        "step": payload["step"],
    }
    metadata_path = args.output_checkpoint.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved {args.output_checkpoint}", flush=True)
    print(f"Metadata {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
