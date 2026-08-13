"""评估统一接口。

支持 LM (perplexity)、CLS (accuracy)、CV (accuracy)、REG (pearson) 任务。
支持 lm_eval harness 下游评测 (HellaSwag, ARC-Easy, PIQA 等)。
"""

import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from experiments.lib.losses import (
    causal_lm_loss,
    extract_logits,
    move_batch_to_device,
    perplexity_from_loss,
)


def evaluate(
    model: nn.Module,
    cached_eval: List[Dict[str, torch.Tensor]],
    task_type: str,
    device: str = "cuda",
) -> Dict[str, float]:
    """在缓存的评估批次上评估模型。

    参数:
        model: PyTorch 模型
        cached_eval: 由 cache_batches 返回的缓存列表
        task_type: 'lm' | 'cls' | 'cv' | 'reg'
        device: 设备

    返回:
        指标字典，如 {'loss': ..., 'perplexity': ...}
    """
    was_training = model.training
    model.eval()
    model = model.to(device)
    try:
        if task_type == "lm":
            return _evaluate_lm(model, cached_eval, device)
        if task_type in ("cls", "cv"):
            return _evaluate_classification(model, cached_eval, device, task_type)
        if task_type == "reg":
            return _evaluate_regression(model, cached_eval, device)
        raise ValueError(f"Unknown task_type: {task_type}")
    finally:
        model.train(was_training)


def _metric_value(metrics: Dict[str, float], key: str, role: str) -> float:
    """Read one finite metric and report malformed evaluation records early."""
    if key not in metrics:
        raise KeyError(f"{role} metrics must contain '{key}'")
    try:
        value = float(metrics[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{role} metric '{key}' must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{role} metric '{key}' must be finite")
    return value


def _relative_drop_pct(delta: float, baseline: float) -> float:
    """Scale a quality delta by the baseline without dividing by zero."""
    if baseline == 0.0:
        if delta == 0.0:
            return 0.0
        raise ValueError("relative quality drop is undefined for a zero baseline")
    return delta / abs(baseline) * 100.0


def compute_quality_drop(
    baseline: Dict[str, float],
    pruned: Dict[str, float],
    task_type: str,
) -> float:
    """计算剪枝后的质量下降比例（百分比）。

    Loss increases and higher-is-better metric decreases are both positive
    drops. A zero baseline is neutral only when the compared metric is also
    zero; non-zero changes fail closed because their relative drop is undefined.
    """
    if task_type == "lm":
        baseline_value = _metric_value(baseline, "loss", "baseline")
        pruned_value = _metric_value(pruned, "loss", "pruned")
        return _relative_drop_pct(pruned_value - baseline_value, baseline_value)
    if task_type in ("cls", "cv"):
        baseline_value = _metric_value(baseline, "accuracy", "baseline")
        pruned_value = _metric_value(pruned, "accuracy", "pruned")
        return _relative_drop_pct(baseline_value - pruned_value, baseline_value)
    if task_type == "reg":
        baseline_value = _metric_value(baseline, "pearson", "baseline")
        pruned_value = _metric_value(pruned, "pearson", "pruned")
        return _relative_drop_pct(baseline_value - pruned_value, baseline_value)
    raise ValueError(f"Unknown task_type: {task_type}")


def _evaluate_lm(model, cached_eval, device) -> Dict[str, float]:
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in cached_eval:
            batch = move_batch_to_device(batch, device)
            loss, supervised_tokens = causal_lm_loss(model, batch, reduction="sum")
            total_loss += loss.item()
            total_tokens += supervised_tokens

    if total_tokens == 0:
        raise ValueError("LM evaluation requires at least one supervised token")
    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "perplexity": perplexity_from_loss(avg_loss),
    }


def _evaluate_classification(model, cached_eval, device, task_type) -> Dict[str, float]:
    if not cached_eval:
        raise ValueError("classification evaluation requires at least one batch")
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in cached_eval:
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"]
            if task_type == "cv":
                outputs = model(batch["images"])
            else:
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )

            logits = extract_logits(outputs)
            loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    return {
        "loss": avg_loss,
        "accuracy": correct / max(total, 1),
    }


def _evaluate_regression(model, cached_eval, device) -> Dict[str, float]:
    if not cached_eval:
        raise ValueError("regression evaluation requires at least one batch")

    total_loss = 0.0
    total_elements = 0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in cached_eval:
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"].float()
            outputs = model(
                batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            logits = extract_logits(outputs)
            preds = logits.squeeze(-1)
            loss = nn.functional.mse_loss(preds, labels, reduction="sum")
            total_loss += loss.item()
            total_elements += labels.numel()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / total_elements
    stacked_preds = torch.cat(all_preds)
    stacked_labels = torch.cat(all_labels)
    if not bool(torch.isfinite(stacked_preds).all().item()) or not bool(
        torch.isfinite(stacked_labels).all().item()
    ):
        raise ValueError("regression predictions and labels must be finite")

    # Pearson correlation
    vp = stacked_preds.to(torch.float64) - stacked_preds.to(torch.float64).mean()
    vl = stacked_labels.to(torch.float64) - stacked_labels.to(torch.float64).mean()
    if vp.norm() == 0 or vl.norm() == 0:
        raise ValueError("Pearson correlation requires non-constant predictions and labels")
    pearson = (vp * vl).sum() / (vp.norm() * vl.norm())
    if not bool(torch.isfinite(pearson).item()):
        raise ValueError("regression Pearson correlation must be finite")

    return {
        "loss": avg_loss,
        "pearson": pearson.item(),
    }


# ============================================================
#  lm_eval harness 下游评测
# ============================================================


def evaluate_downstream(
    model: nn.Module,
    tokenizer_name: str,
    tasks: Optional[List[str]] = None,
    device: str = "cuda",
    batch_size: int = 8,
    num_fewshot: int = 0,
) -> Dict[str, float]:
    """用 lm_eval harness 评估下游任务。

    参数:
        model: HuggingFace 因果语言模型
        tokenizer_name: tokenizer 名称（如 'EleutherAI/pythia-410m'）
        tasks: 评估任务列表（默认 ['hellaswag', 'arc_easy', 'piqa']）
        device: 设备
        batch_size: 评估批次大小
        num_fewshot: few-shot 数量

    返回:
        {task_name: accuracy, ..., 'avg_acc': ...}
    """
    if tasks is None:
        tasks = ["hellaswag", "arc_easy", "piqa"]
    if (
        not tasks
        or any(not isinstance(task, str) or not task.strip() for task in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        raise ValueError("tasks must be a non-empty list of unique non-empty names")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_fewshot, bool) or not isinstance(num_fewshot, int) or num_fewshot < 0:
        raise ValueError("num_fewshot must be a non-negative integer")

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError("需要安装 lm_eval: pip install lm_eval>=0.4.0")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wrap model for lm_eval
    hf_model = model.model if hasattr(model, "model") else model
    lm = HFLM(
        pretrained=hf_model,
        tokenizer=tokenizer,
        device=str(device),
        batch_size=batch_size,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    if not isinstance(results, Mapping) or not isinstance(results.get("results"), Mapping):
        raise ValueError("lm_eval response must contain a results object")
    task_results = results["results"]
    metrics: Dict[str, float] = {}
    acc_sum = 0.0
    for task in tasks:
        task_result = task_results.get(task)
        if not isinstance(task_result, Mapping):
            raise ValueError(f"lm_eval response is missing results for task {task!r}")
        metric_key = next(
            (key for key in ("acc,none", "acc_norm,none") if key in task_result),
            None,
        )
        if metric_key is None:
            raise ValueError(f"lm_eval response for task {task!r} has no accuracy metric")
        raw_acc = task_result[metric_key]
        if isinstance(raw_acc, bool):
            raise ValueError(f"lm_eval accuracy for task {task!r} must be numeric")
        try:
            acc = float(raw_acc)
        except (TypeError, ValueError) as error:
            raise ValueError(f"lm_eval accuracy for task {task!r} must be numeric") from error
        if not math.isfinite(acc) or not 0.0 <= acc <= 1.0:
            raise ValueError(f"lm_eval accuracy for task {task!r} must be finite and in [0, 1]")
        metrics[task] = acc
        acc_sum += acc

    metrics["avg_acc"] = acc_sum / len(tasks)
    return metrics
