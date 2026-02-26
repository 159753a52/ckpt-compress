"""
AdamPrune: 基于Adam二阶矩的理论剪枝方法。

利用 Adam 优化器的二阶矩作为 Hessian 对角线的代理，
通过数学公式直接计算给定损失容忍度下的最优剪枝比例。

核心流程:
1. 计算重要性: s_i = -g_i·θ_i + α·v_i·θ_i² (Adam近似) 或 s_i = -g_i·θ_i + 0.5·θ_i·(H·θ)_i (HVP精确)
2. 分布拟合: 每层拟合韦伯分布
3. 拉格朗日优化: 二分搜索η，计算各层剪枝比例
4. 分层剪枝: 应用剪枝比例
"""

from .importance import (
    compute_importance_scores,
    compute_importance_scores_hvp,
    compute_importance_scores_hvp_memory_efficient,
    compute_hvp,
    compute_hvp_batched,
)
from .distribution import fit_exponential_distribution, analyze_distribution, get_best_fit
from .distribution_fast import fit_distributions_per_layer_fast, fit_weibull_linear_regression
from .layer_pruning import (
    LayerPruningOptimizer,
    prune_by_layer_sparsity,
    prune_by_global_sparsity_fast,
    compute_layer_sparsities_from_global,
)
from .adaptive_pruning import (
    fit_distributions_per_layer,
    build_layer_dist_params,
    filter_prunable_layers,
)

__all__ = [
    # 重要性计算 (Adam 近似)
    "compute_importance_scores",
    # 重要性计算 (HVP 精确)
    "compute_importance_scores_hvp",
    "compute_importance_scores_hvp_memory_efficient",
    "compute_hvp",
    "compute_hvp_batched",
    # 分布拟合
    "fit_exponential_distribution",
    "analyze_distribution",
    "get_best_fit",
    "fit_distributions_per_layer_fast",
    "fit_weibull_linear_regression",
    "fit_distributions_per_layer",
    "build_layer_dist_params",
    "filter_prunable_layers",
    # 分层剪枝
    "LayerPruningOptimizer",
    "prune_by_layer_sparsity",
    "prune_by_global_sparsity_fast",
    "compute_layer_sparsities_from_global",
]
