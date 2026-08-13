"""共享实验流水线函数。

将各 run_*.py 脚本中重复的 模型加载 → 数据缓存 → 基线评估 → 得分计算
流程提取到此处。
"""

import copy
from argparse import ArgumentTypeError

import torch

from experiments.lib.args import device_name, nonnegative_int, positive_int, unit_interval_float
from experiments.lib.data import cache_batches, get_data_loaders
from experiments.lib.evaluation import evaluate
from experiments.lib.importance_compare.scoring import compute_scores_by_method
from experiments.lib.losses import compute_task_loss
from experiments.lib.models import load_model


def _validate_runtime_args(args, cache_eval=True) -> None:
    """Validate direct API callers before loading models or datasets."""
    validators = {
        "batch_size": positive_int,
        "seq_length": positive_int,
        "num_workers": nonnegative_int,
        "num_steps": positive_int,
        "hvp_batches": positive_int,
        "chunk_size": positive_int,
        "alpha": unit_interval_float,
        "device": device_name,
    }
    if cache_eval:
        validators["eval_batches"] = positive_int
    defaults = {
        "num_workers": 0,
        "num_steps": 100,
        "hvp_batches": 8,
        "chunk_size": 10,
        "alpha": 0.5,
        "device": "cuda",
        "eval_batches": 20,
    }
    for name, validator in validators.items():
        value = getattr(args, name, defaults.get(name))
        if value is None:
            raise ValueError(f"{name} is required")
        try:
            validator(str(value))
        except (ArgumentTypeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid {name}: {value!r}") from exc


def setup_model_and_data(args, cache_eval=True):
    """加载模型 & 数据，缓存 batch。

    Returns:
        model: 加载到 CPU 的模型
        cached_train: 缓存的训练 batch 列表
        cached_eval: 缓存的评估 batch 列表（cache_eval=False 时为 None）
        task_type: 'lm' / 'cls' / 'cv' / 'reg'
        train_loader: 原始训练 DataLoader（部分脚本需要）
        val_loader: 原始验证 DataLoader（部分脚本需要）
    """
    _validate_runtime_args(args, cache_eval=cache_eval)
    print(f"\n[Setup] 加载模型: {args.model}")
    model, model_family = load_model(
        args.model,
        pretrained=True,
        checkpoint_path=args.checkpoint,
        device="cpu",
        dataset_name=args.dataset,
    )

    print(f"[Setup] 加载数据: {args.dataset}")
    train_loader, val_loader, task_type = get_data_loaders(
        args.model,
        args.dataset,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=getattr(args, "num_workers", 0),
        data_dir=getattr(args, "data_dir", "./data"),
    )

    num_steps = getattr(args, "num_steps", 100)
    eval_batches = getattr(args, "eval_batches", 20)

    print(f"[Setup] 缓存批次 (train={num_steps}, eval={eval_batches})...")
    cached_train = cache_batches(train_loader, num_steps, task_type)
    cached_eval = cache_batches(val_loader, eval_batches, task_type) if cache_eval else None

    # 将 model_family 保存到 args 中，供后续 precompute_scores 使用
    args.model_family = model_family

    return model, cached_train, cached_eval, task_type, train_loader, val_loader


def evaluate_baseline(model, cached_eval, task_type, device):
    """评估未剪枝模型的 baseline 性能。

    Returns:
        baseline_metrics: dict, 例如 {'loss': 3.12, 'perplexity': 22.6}
    """
    print("[Setup] 评估 baseline ...")
    model_eval = copy.deepcopy(model).to(device)
    baseline = evaluate(model_eval, cached_eval, task_type, device)
    print(f"  Baseline: {baseline}")
    del model_eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return baseline


def precompute_scores(model, cached_train, task_type, methods, args, **kwargs):
    """预计算一组 importance method 的得分。

    Args:
        model: CPU 上的模型（会 deepcopy）
        methods: list[str], 如 ['magnitude', 'first-order', 'second-order-hvp']
        args: 需包含 device, alpha, hvp_batches; 可选 hvp_mode, chunk_size
        **kwargs: 透传给 compute_scores_by_method（如 reference_weights, grad_batches_first_order）

    Returns:
        score_cache: dict, {method_name: {param_name: score_tensor}}
    """
    model_for_scoring = copy.deepcopy(model).to(args.device)
    score_cache = compute_scores_by_method(
        model_for_scoring,
        cached_train,
        task_type,
        methods=methods,
        alpha=getattr(args, "alpha", 0.5),
        hvp_batches=getattr(args, "hvp_batches", 8),
        hvp_mode=getattr(args, "hvp_mode", "full"),
        chunk_size=getattr(args, "chunk_size", 10),
        model_family=getattr(args, "model_family", "gpt2"),
        **kwargs,
    )
    del model_for_scoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score_cache


def get_metric_key(task_type):
    """根据任务类型返回主评估指标名。"""
    if task_type == "lm":
        return "perplexity"
    elif task_type == "reg":
        return "pearson"
    return "accuracy"


def train_one_step(model, optimizer, batch, task_type, device):
    """训练一步，返回 loss。"""
    model.train()
    loss = compute_task_loss(model, batch, task_type, device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
