"""评估统一接口。

支持 LM (perplexity)、CLS (accuracy)、CV (accuracy)、REG (pearson) 任务。
支持 lm_eval harness 下游评测 (HellaSwag, ARC-Easy, PIQA 等)。
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional


def evaluate(
    model: nn.Module,
    cached_eval: List[Dict[str, torch.Tensor]],
    task_type: str,
    device: str = 'cuda',
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
    model.eval()
    model = model.to(device)

    if task_type == 'lm':
        return _evaluate_lm(model, cached_eval, device)
    elif task_type in ('cls', 'cv'):
        return _evaluate_classification(model, cached_eval, device, task_type)
    elif task_type == 'reg':
        return _evaluate_regression(model, cached_eval, device)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def compute_quality_drop(baseline: Dict[str, float], pruned: Dict[str, float], task_type: str) -> float:
    """计算剪枝后的质量下降比例（百分比）。"""
    if task_type == 'lm':
        return (pruned['loss'] - baseline['loss']) / baseline['loss'] * 100
    elif task_type in ('cls', 'cv'):
        return (baseline['accuracy'] - pruned.get('accuracy', 0)) / baseline['accuracy'] * 100
    elif task_type == 'reg':
        return (baseline.get('pearson', 1) - pruned.get('pearson', 0)) / max(baseline.get('pearson', 1), 1e-8) * 100
    return 0.0


def _evaluate_lm(model, cached_eval, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in cached_eval:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1))
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / max(count, 1)
    return {
        'loss': avg_loss,
        'perplexity': math.exp(min(avg_loss, 20)),  # 防止 overflow
    }


def _evaluate_classification(model, cached_eval, device, task_type):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in cached_eval:
            if task_type == 'cv':
                inputs = batch['images'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(inputs)
            else:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attn = batch.get('attention_mask')
                if attn is not None:
                    attn = attn.to(device)
                outputs = model(input_ids, attention_mask=attn)

            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(len(cached_eval), 1)
    return {
        'loss': avg_loss,
        'accuracy': correct / max(total, 1),
    }


def _evaluate_regression(model, cached_eval, device):
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in cached_eval:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device).float()
            attn = batch.get('attention_mask')
            if attn is not None:
                attn = attn.to(device)
            outputs = model(input_ids, attention_mask=attn)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = logits.squeeze(-1)
            loss = nn.MSELoss()(preds, labels)
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / max(len(cached_eval), 1)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Pearson correlation
    vp = all_preds - all_preds.mean()
    vl = all_labels - all_labels.mean()
    pearson = (vp * vl).sum() / (vp.norm() * vl.norm() + 1e-8)

    return {
        'loss': avg_loss,
        'pearson': pearson.item(),
    }


# ============================================================
#  lm_eval harness 下游评测
# ============================================================

def evaluate_downstream(
    model: nn.Module,
    tokenizer_name: str,
    tasks: List[str] = None,
    device: str = 'cuda',
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
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError(
            "需要安装 lm_eval: pip install lm_eval>=0.4.0"
        )

    if tasks is None:
        tasks = ['hellaswag', 'arc_easy', 'piqa']

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wrap model for lm_eval
    hf_model = model.model if hasattr(model, 'model') else model
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

    metrics = {}
    acc_sum = 0.0
    for task in tasks:
        task_result = results['results'].get(task, {})
        acc = task_result.get('acc,none', task_result.get('acc_norm,none', 0.0))
        metrics[task] = acc
        acc_sum += acc

    metrics['avg_acc'] = acc_sum / max(len(tasks), 1)
    return metrics
