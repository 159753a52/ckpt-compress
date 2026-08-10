"""Shared task-loss factories used by experiment scoring and training code."""

from typing import Callable

import torch
import torch.nn as nn


SUPPORTED_TASK_TYPES = frozenset({"lm", "cls", "cv", "reg"})


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


__all__ = ["SUPPORTED_TASK_TYPES", "make_task_loss"]
