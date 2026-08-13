"""Runtime, checkpoint, data, and evaluation helpers for residual experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, cast

import numpy as np
import torch
import torch.nn as nn

from experiments.lib.evaluation import evaluate
from experiments.lib.losses import (
    causal_lm_loss,
    compute_task_loss,
    move_batch_to_device,
    perplexity_from_loss,
)


@dataclass(frozen=True)
class LoadedTrainingCheckpoint:
    """Model and optimizer states decoded together from one trusted checkpoint."""

    model_state: Dict[str, torch.Tensor]
    optimizer_state: Dict
    step: int


def configure_hf_offline() -> None:
    """Configure deterministic local-only model and tokenizer loading."""
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def empty_device_cache(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def reset_peak_memory(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def peak_memory_bytes(device: str) -> int:
    if not device.startswith("cuda"):
        return 0
    return int(torch.cuda.max_memory_allocated(torch.device(device)))


def synchronize_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def set_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint_payload(path: Path) -> object:
    # Experiments only accept trusted local checkpoints because optimizer payloads
    # may contain Python objects unsupported by weights_only=True.
    return torch.load(path, map_location="cpu", weights_only=False)


def _checkpoint_state_from_payload(
    payload: object,
    path: Path,
) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload in {path}: {type(payload)}")
    for key in ("model_state_dict", "state_dict", "model"):
        if key in payload and isinstance(payload[key], dict):
            state = payload[key]
            break
    else:
        state = (
            payload
            if payload and all(torch.is_tensor(value) for value in payload.values())
            else None
        )
    if not isinstance(state, dict) or not state:
        raise KeyError(f"No model state dict found in {path}")
    if not all(isinstance(name, str) and torch.is_tensor(value) for name, value in state.items()):
        raise TypeError(f"Model state dict in {path} must map names to tensors")
    return state


def _checkpoint_optimizer_state_from_payload(payload: object, path: Path) -> Dict:
    if not isinstance(payload, dict) or "optimizer_state_dict" not in payload:
        raise KeyError(f"No optimizer state dict found in {path}")
    state = payload["optimizer_state_dict"]
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("state"), Mapping)
        or not isinstance(state.get("param_groups"), list)
        or not all(isinstance(value, Mapping) for value in state["state"].values())
        or not all(isinstance(group, Mapping) for group in state["param_groups"])
    ):
        raise TypeError(f"Optimizer state dict in {path} has an invalid structure")
    return state


def _checkpoint_step(payload: Mapping, path: Path) -> int:
    raw = payload.get("step", 0)
    if isinstance(raw, bool):
        raise ValueError(f"Checkpoint step in {path} must be a non-negative integer")
    if isinstance(raw, str):
        if not raw.isdecimal():
            raise ValueError(f"Checkpoint step in {path} must be a non-negative integer")
        value = int(raw)
    elif isinstance(raw, int):
        value = raw
    else:
        raise ValueError(f"Checkpoint step in {path} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"Checkpoint step in {path} must be a non-negative integer")
    return value


def checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    return _checkpoint_state_from_payload(_load_checkpoint_payload(path), path)


def checkpoint_optimizer_state(path: Path) -> Dict:
    return _checkpoint_optimizer_state_from_payload(
        _load_checkpoint_payload(path),
        path,
    )


def load_training_checkpoint(path: Path) -> LoadedTrainingCheckpoint:
    """Load model, optimizer, and step metadata with one CPU deserialization."""
    payload = _load_checkpoint_payload(path)
    model_state = _checkpoint_state_from_payload(payload, path)
    optimizer_state = _checkpoint_optimizer_state_from_payload(payload, path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload in {path}: {type(payload)}")
    step = _checkpoint_step(payload, path)
    return LoadedTrainingCheckpoint(model_state, optimizer_state, step)


def optimizer_state_to_cpu(state: Mapping) -> Dict:
    """Clone an optimizer state dict to CPU without a transient GPU deepcopy."""
    result = {"state": {}, "param_groups": copy.deepcopy(state["param_groups"])}
    for parameter_id, parameter_state in state["state"].items():
        result["state"][parameter_id] = {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in parameter_state.items()
        }
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_token_batches(
    path: Path,
    tokenizer,
    batch_size: int,
    seq_length: int,
    num_batches: int,
    batch_offset: int = 0,
) -> List[Dict[str, torch.Tensor]]:
    for name, value in (
        ("batch_size", batch_size),
        ("seq_length", seq_length),
        ("num_batches", num_batches),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value}")
    if isinstance(batch_offset, bool) or not isinstance(batch_offset, int) or batch_offset < 0:
        raise ValueError(f"batch_offset must be a non-negative integer, got {batch_offset}")

    skip = batch_size * seq_length * batch_offset
    needed = batch_size * seq_length * num_batches
    required = skip + needed
    tokens: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tokens.extend(tokenizer.encode(line))
            if len(tokens) >= required:
                break
    if len(tokens) < required:
        raise RuntimeError(f"Only found {len(tokens)} tokens in {path}, need {required}")

    batches = []
    offset = skip
    for _ in range(num_batches):
        size = batch_size * seq_length
        input_ids = torch.tensor(tokens[offset : offset + size], dtype=torch.long)
        input_ids = input_ids.view(batch_size, seq_length)
        batches.append({"input_ids": input_ids, "labels": input_ids.clone()})
        offset += size
    return batches


def batch_hash(batches: Sequence[Mapping[str, torch.Tensor]]) -> str:
    """Hash arbitrary task batches without depending on one input schema."""
    digest = hashlib.sha256()
    legacy_lm_schema = all(
        set(batch) == {"input_ids", "labels"}
        and torch.is_tensor(batch["input_ids"])
        and torch.is_tensor(batch["labels"])
        and torch.equal(batch["input_ids"], batch["labels"])
        for batch in batches
    )
    for batch in batches:
        if legacy_lm_schema:
            digest.update(batch["input_ids"].detach().cpu().contiguous().numpy().tobytes())
            continue
        tensor_items = sorted(
            (key, value) for key, value in batch.items() if torch.is_tensor(value)
        )
        if not tensor_items:
            raise ValueError("Every batch must contain at least one tensor")
        for key, value in tensor_items:
            cpu_value = value.detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(cpu_value.dtype).encode("ascii"))
            digest.update(str(tuple(cpu_value.shape)).encode("ascii"))
            digest.update(cpu_value.numpy().tobytes())
    return digest.hexdigest()


def lm_loss(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    device: str,
) -> torch.Tensor:
    batch_on_device = move_batch_to_device(batch, device)
    loss, _ = causal_lm_loss(model, batch_on_device)
    return loss


def task_loss(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    task_type: str,
    device: str,
) -> torch.Tensor:
    """Compute a task loss through the shared experiment contract."""
    return compute_task_loss(model, batch, task_type, device)


def evaluate_lm(
    model: nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    device: str,
) -> Dict[str, float]:
    if not batches:
        raise ValueError("batches must contain at least one evaluation batch")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for batch in batches:
                batch_on_device = move_batch_to_device(batch, device)
                loss, supervised_tokens = causal_lm_loss(
                    model,
                    batch_on_device,
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += supervised_tokens
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("LM evaluation requires at least one supervised token")
    average = total_loss / total_tokens
    return {
        "loss": average,
        "perplexity": perplexity_from_loss(average),
        "seconds": time.perf_counter() - started,
        "batches": len(batches),
        "tokens": total_tokens,
    }


def evaluate_task(
    model: nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
    task_type: str,
    device: str,
) -> dict[str, float | int]:
    """Evaluate any task supported by the shared experiment library."""
    if not batches:
        raise ValueError("batches must contain at least one evaluation batch")
    started = time.perf_counter()
    eval_batches = [cast(Dict[str, torch.Tensor], dict(batch)) for batch in batches]
    metrics: dict[str, float | int] = dict(evaluate(model, eval_batches, task_type, device))
    metrics["seconds"] = time.perf_counter() - started
    metrics["batches"] = len(batches)
    metrics["examples"] = sum(
        next(value for value in batch.values() if torch.is_tensor(value)).shape[0]
        for batch in batches
    )
    return metrics


__all__ = [
    "LoadedTrainingCheckpoint",
    "batch_hash",
    "checkpoint_optimizer_state",
    "checkpoint_state",
    "configure_hf_offline",
    "empty_device_cache",
    "evaluate_lm",
    "evaluate_task",
    "lm_loss",
    "load_token_batches",
    "load_training_checkpoint",
    "optimizer_state_to_cpu",
    "peak_memory_bytes",
    "reset_peak_memory",
    "set_seed",
    "sha256_file",
    "synchronize_device",
    "task_loss",
    "write_json",
]
