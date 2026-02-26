"""
Adaptive pruning helper tests.
"""

import os
from typing import Dict

import pytest
import torch


def test_is_bias_param():
    """bias 参数应识别为 True。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import is_bias_param

    assert is_bias_param("transformer.h.0.attn.c_attn.bias") is True
    assert is_bias_param("transformer.h.0.attn.c_attn.weight") is False
    assert is_bias_param("layer.bias") is True


def test_filter_prunable_layers_preserve_bias():
    """保留 bias 时应过滤掉 bias。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import filter_prunable_layers

    layer_scores = {
        "layer.weight": torch.ones(4),
        "layer.bias": torch.ones(2),
    }

    prunable, frozen = filter_prunable_layers(layer_scores, preserve_bias=True)

    assert "layer.weight" in prunable
    assert "layer.bias" not in prunable
    assert frozen == ["layer.bias"]


def test_filter_prunable_layers_keep_bias():
    """不保留 bias 时不应过滤。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import filter_prunable_layers

    layer_scores = {
        "layer.weight": torch.ones(4),
        "layer.bias": torch.ones(2),
    }

    prunable, frozen = filter_prunable_layers(layer_scores, preserve_bias=False)

    assert set(prunable.keys()) == set(layer_scores.keys())
    assert frozen == []


def test_fit_distributions_per_layer_small_sample():
    """样本过少时使用原始scipy方法应返回 unknown。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import (
        fit_distributions_per_layer,
    )

    layer_scores = {
        "layer.weight": torch.tensor([0.1, 0.2, 0.3]),
    }

    # 使用原始 scipy 方法（use_fast=False），小样本会返回 unknown
    results = fit_distributions_per_layer(layer_scores, use_fast=False)
    assert results["layer.weight"]["name"] == "unknown"


def test_fit_distributions_per_layer_fast_small_sample():
    """快速方法可以处理小样本。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import (
        fit_distributions_per_layer,
    )

    layer_scores = {
        "layer.weight": torch.tensor([0.1, 0.2, 0.3]),
    }

    # 快速方法可以处理小样本，返回 weibull
    results = fit_distributions_per_layer(layer_scores, use_fast=True)
    assert results["layer.weight"]["name"] == "weibull"


def test_build_layer_dist_params_with_valid_fit():
    """应正确转换分布参数。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import build_layer_dist_params

    layer_scores = {"layer.weight": torch.ones(10)}
    fit_results: Dict[str, Dict] = {
        "layer.weight": {
            "name": "weibull",
            "params": (2.0, 0.0, 0.5),
        }
    }

    params = build_layer_dist_params(layer_scores, fit_results)

    assert params["layer.weight"]["N"] == 10
    assert params["layer.weight"]["dist_type"] == "weibull"
    assert params["layer.weight"]["k"] == 2.0
    assert params["layer.weight"]["beta"] == 0.5
    assert params["layer.weight"]["loc"] == 0.0


def test_build_layer_dist_params_fallback_on_missing():
    """缺少拟合信息时应使用默认参数。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import build_layer_dist_params

    layer_scores = {"layer.weight": torch.ones(5)}
    params = build_layer_dist_params(layer_scores, {})

    assert params["layer.weight"]["dist_type"] == "weibull"
    assert params["layer.weight"]["k"] == 1.0
    assert params["layer.weight"]["beta"] == 0.1
    assert params["layer.weight"]["loc"] == 0.0


def test_apply_prune_ratio_overrides():
    """应为 bias 填充 0 剪枝比例。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import (
        apply_prune_ratio_overrides,
    )

    layer_scores = {
        "layer.weight": torch.ones(4),
        "layer.bias": torch.ones(2),
    }
    prune_ratios = {"layer.weight": 0.2}

    merged = apply_prune_ratio_overrides(
        prune_ratios, layer_scores, preserve_bias=True
    )

    assert merged["layer.weight"] == 0.2
    assert merged["layer.bias"] == 0.0


def test_compute_target_pruned_score_budget():
    """应根据幂律参数计算 p*。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import (
        compute_target_pruned_score_budget,
    )

    budget = compute_target_pruned_score_budget(
        target_rel_loss=0.1,
        baseline_loss=2.0,
        powerlaw_params={"a": 2.0, "b": 1.0},
    )

    assert abs(budget - 0.1) < 1e-6


def test_collect_calibration_points_skips_invalid():
    """应跳过缺失字段的数据点。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import (
        collect_calibration_points,
    )

    results = [
        {"sum_pruned_scores": 0.2, "actual_loss_increase": 0.02},
        {"sum_pruned_scores": 0.1},
    ]

    xs, ys = collect_calibration_points(results, baseline_loss=2.0)

    assert xs == [0.1]
    assert ys == [0.01]


def test_adjust_epsilon_for_target_reduces():
    """当实际误差过高时应降低 epsilon。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import (
        adjust_epsilon_for_target,
    )

    def eval_fn(epsilon: float) -> float:
        return epsilon * 2.0

    epsilon, history = adjust_epsilon_for_target(
        initial_epsilon=0.1,
        target_rel_loss=0.05,
        evaluate_fn=eval_fn,
        max_iters=3,
    )

    assert epsilon <= 0.025
    assert history


def test_setup_hf_cache_sets_env(monkeypatch):
    """应设置 HF 缓存环境变量。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import setup_hf_cache

    cache_dir = "/tmp/hf_cache_test"
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)

    setup_hf_cache(cache_dir)

    assert os.environ["HF_HOME"] == cache_dir
    assert os.environ["TRANSFORMERS_CACHE"] == cache_dir
    assert os.environ["HF_DATASETS_CACHE"] == cache_dir


def test_get_block_number():
    """应正确解析 block 编号。"""
    from ckpt_compress.methods.adam_prune.adaptive_pruning import get_block_number

    assert get_block_number("transformer.h.3.mlp.c_fc.weight") == 3
    assert get_block_number("transformer.wte.weight") == -1
