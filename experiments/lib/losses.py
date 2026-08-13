"""Shared task-loss factories used by experiment scoring and training code."""

import math
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Callable

import torch
import torch.nn as nn

SUPPORTED_TASK_TYPES = frozenset({"lm", "cls", "cv", "reg"})
TRAINING_TASK_TYPES = SUPPORTED_TASK_TYPES
CAUSAL_LM_IGNORE_INDEX = -100


def extract_logits(outputs: object) -> torch.Tensor:
    """Return logits from either a tensor output or a model output object."""
    return outputs.logits if hasattr(outputs, "logits") else outputs


def perplexity_from_loss(loss: float) -> float:
    """Convert a finite non-negative mean NLL to perplexity without clipping."""
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise TypeError(f"loss must be numeric, got {type(loss).__name__}")
    value = float(loss)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"loss must be finite and non-negative, got {loss}")
    try:
        perplexity = math.exp(value)
    except OverflowError as exc:
        raise ValueError(f"loss is too large for finite perplexity: {loss}") from exc
    if not math.isfinite(perplexity):
        raise ValueError(f"loss produced non-finite perplexity: {loss}")
    return perplexity


def causal_lm_loss(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    reduction: str = "mean",
) -> tuple[torch.Tensor, int]:
    """Compute shifted causal-LM loss and return its supervised-token count."""
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported causal-LM reduction: {reduction!r}")
    missing = [key for key in ("input_ids", "labels") if key not in batch]
    if missing:
        raise KeyError(f"causal-LM batch is missing fields: {missing}")

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    if input_ids.dtype != torch.long or labels.dtype != torch.long:
        raise TypeError("causal-LM input_ids and labels must have dtype torch.long")
    if input_ids.shape != labels.shape:
        raise ValueError(
            "causal-LM labels must have the same shape as input_ids: "
            f"{tuple(labels.shape)} != {tuple(input_ids.shape)}"
        )
    if input_ids.ndim < 2 or input_ids.shape[-1] < 2:
        raise ValueError("causal-LM batches need sequence length >= 2")

    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("causal-LM attention_mask must match input_ids shape")
        if attention_mask.device != input_ids.device:
            raise ValueError("causal-LM attention_mask and input_ids must share a device")
        if attention_mask.is_complex() or not bool(
            torch.all((attention_mask == 0) | (attention_mask == 1)).item()
        ):
            raise ValueError("causal-LM attention_mask must contain only 0 and 1")
    if attention_mask is None:
        outputs = model(input_ids)
    else:
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = extract_logits(outputs)
    if not logits.is_floating_point():
        raise TypeError("causal-LM logits must be floating-point")
    if logits.shape[:-1] != input_ids.shape:
        raise ValueError(
            "causal-LM logits must match the input batch and sequence dimensions: "
            f"{tuple(logits.shape[:-1])} != {tuple(input_ids.shape)}"
        )

    shift_logits = logits[..., :-1, :].contiguous()
    effective_labels = labels.clone()
    if attention_mask is not None:
        effective_labels.masked_fill_(attention_mask == 0, CAUSAL_LM_IGNORE_INDEX)
    shift_labels = effective_labels[..., 1:].contiguous()
    supervised_tokens = int((shift_labels != CAUSAL_LM_IGNORE_INDEX).sum().item())
    if supervised_tokens == 0:
        raise ValueError("causal-LM batch contains no supervised tokens")
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=CAUSAL_LM_IGNORE_INDEX,
        reduction=reduction,
    )
    return loss, supervised_tokens


@lru_cache(maxsize=None)
def make_task_loss(task_type: str) -> Callable:
    """Build a ``loss(model, batch) -> scalar`` function for a task type.

    Batches are expected to already be on the model device. Regression uses
    the same MSE objective as evaluation and the GLUE fine-tuning entrypoint.
    """
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unknown task_type: {task_type}. " f"Available: {sorted(SUPPORTED_TASK_TYPES)}"
        )

    criterion = nn.CrossEntropyLoss()

    def lm_loss(model, batch):
        loss, _ = causal_lm_loss(model, batch)
        return loss

    def cls_loss(model, batch):
        attention_mask = batch.get("attention_mask")
        outputs = model(batch["input_ids"], attention_mask=attention_mask)
        return criterion(extract_logits(outputs), batch["labels"])

    def cv_loss(model, batch):
        outputs = model(batch["images"])
        return criterion(extract_logits(outputs), batch["labels"])

    def reg_loss(model, batch):
        labels = batch["labels"]
        if not labels.is_floating_point():
            raise TypeError("regression labels must be floating-point")
        outputs = model(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        predictions = extract_logits(outputs)
        if predictions.ndim == labels.ndim + 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)
        if predictions.shape != labels.shape:
            raise ValueError(
                "regression predictions must match label shape: "
                f"{tuple(predictions.shape)} != {tuple(labels.shape)}"
            )
        if not predictions.is_floating_point():
            raise TypeError("regression predictions must be floating-point")
        return nn.functional.mse_loss(predictions, labels)

    return {"lm": lm_loss, "cls": cls_loss, "cv": cv_loss, "reg": reg_loss}[task_type]


def move_batch_to_device(batch: Any, device: Any) -> Any:
    """Move tensor values in mapping/sequence batches to ``device``."""
    if isinstance(batch, Mapping):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(
            value.to(device) if isinstance(value, torch.Tensor) else value for value in batch
        )
    if isinstance(batch, list):
        return [value.to(device) if isinstance(value, torch.Tensor) else value for value in batch]
    raise TypeError(
        "batch must be a mapping, tuple, or list containing tensors; " f"got {type(batch).__name__}"
    )


def compute_task_loss(
    model: nn.Module,
    batch: Any,
    task_type: str,
    device: Any,
) -> torch.Tensor:
    """Move a training batch and compute its supported task loss.

    CV tuple batches from legacy DataLoaders are normalized to the mapping
    schema consumed by :func:`make_task_loss`.
    """
    if task_type not in TRAINING_TASK_TYPES:
        raise ValueError(
            f"Unknown training task_type: {task_type}. " f"Available: {sorted(TRAINING_TASK_TYPES)}"
        )

    if task_type == "cv" and isinstance(batch, (tuple, list)):
        if len(batch) != 2:
            raise ValueError("CV tuple/list batches must contain exactly (images, labels)")
        batch = {"images": batch[0], "labels": batch[1]}
    elif not isinstance(batch, Mapping):
        raise TypeError(f"{task_type} batches must be mappings, got {type(batch).__name__}")

    batch_on_device = move_batch_to_device(batch, device)
    return make_task_loss(task_type)(model, batch_on_device)


__all__ = [
    "CAUSAL_LM_IGNORE_INDEX",
    "SUPPORTED_TASK_TYPES",
    "TRAINING_TASK_TYPES",
    "causal_lm_loss",
    "compute_task_loss",
    "extract_logits",
    "make_task_loss",
    "move_batch_to_device",
    "perplexity_from_loss",
]
