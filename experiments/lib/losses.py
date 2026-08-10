"""Shared task-loss factories used by experiment scoring and training code."""

from collections.abc import Mapping
from typing import Any, Callable

import torch
import torch.nn as nn


SUPPORTED_TASK_TYPES = frozenset({"lm", "cls", "cv", "reg"})
TRAINING_TASK_TYPES = frozenset({"lm", "cls", "cv"})


def _extract_logits(outputs: object) -> torch.Tensor:
    """Return logits from either a tensor output or a model output object."""
    return outputs.logits if hasattr(outputs, "logits") else outputs


def make_task_loss(task_type: str) -> Callable:
    """Build a ``loss(model, batch) -> scalar`` function for a task type.

    Batches are expected to already be on the model device.  The ``reg``
    mapping intentionally retains the historical classification-loss contract
    used by importance scoring; regression evaluation remains MSE/Pearson.
    """
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Available: {sorted(SUPPORTED_TASK_TYPES)}"
        )

    criterion = nn.CrossEntropyLoss()

    def lm_loss(model, batch):
        outputs = model(batch["input_ids"])
        logits = _extract_logits(outputs)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        return criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def cls_loss(model, batch):
        attention_mask = batch.get("attention_mask")
        outputs = model(batch["input_ids"], attention_mask=attention_mask)
        return criterion(_extract_logits(outputs), batch["labels"])

    def cv_loss(model, batch):
        outputs = model(batch["images"])
        return criterion(_extract_logits(outputs), batch["labels"])

    # Preserve the historical scoring behavior until scoring and regression
    # evaluation are migrated to an explicitly shared MSE contract.
    return {"lm": lm_loss, "cls": cls_loss, "cv": cv_loss, "reg": cls_loss}[task_type]


def move_batch_to_device(batch: Any, device: Any) -> Any:
    """Move tensor values in mapping/sequence batches to ``device``."""
    if isinstance(batch, Mapping):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(
            value.to(device) if isinstance(value, torch.Tensor) else value
            for value in batch
        )
    if isinstance(batch, list):
        return [
            value.to(device) if isinstance(value, torch.Tensor) else value
            for value in batch
        ]
    raise TypeError(
        "batch must be a mapping, tuple, or list containing tensors; "
        f"got {type(batch).__name__}"
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
            f"Unknown training task_type: {task_type}. "
            f"Available: {sorted(TRAINING_TASK_TYPES)}"
        )

    if task_type == "cv" and isinstance(batch, (tuple, list)):
        if len(batch) != 2:
            raise ValueError(
                "CV tuple/list batches must contain exactly (images, labels)"
            )
        batch = {"images": batch[0], "labels": batch[1]}
    elif not isinstance(batch, Mapping):
        raise TypeError(
            f"{task_type} batches must be mappings, got {type(batch).__name__}"
        )

    batch_on_device = move_batch_to_device(batch, device)
    return make_task_loss(task_type)(model, batch_on_device)


__all__ = [
    "SUPPORTED_TASK_TYPES",
    "TRAINING_TASK_TYPES",
    "compute_task_loss",
    "make_task_loss",
    "move_batch_to_device",
]
