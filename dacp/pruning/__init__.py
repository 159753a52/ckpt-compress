"""统一剪枝框架。

提供统一的重要性得分计算、剪枝率分配和剪枝执行接口。
支持多种方法的公平对比。

用法:
    from dacp.pruning import Pruner
    
    pruner = Pruner(importance='first-order', allocation='gamma-adaptive')
    scores = pruner.compute_scores(weights, gradients)
    layer_ratios = pruner.compute_layer_ratios(scores, global_ratio=0.5)
    model, masks, actual_ratio = pruner.prune(model, scores, layer_ratios, device)
"""

from .importance import (
    ImportanceScorer,
    MagnitudeScorer,
    FirstOrderScorer,
    ResidualMagnitudeScorer,
    get_importance_scorer,
    list_importance_methods,
    IMPORTANCE_REGISTRY,
    combine_scores_2d_with_protection,
    apply_magnitude_protection,
)
from .allocation import (
    AllocationStrategy,
    UniformAllocation,
    GammaAdaptiveAllocation,
    WeibullAdaptiveAllocation,
    get_allocation_strategy,
    list_allocation_strategies,
    ALLOCATION_REGISTRY,
)
from .pruner import (
    Pruner,
    filter_prunable_params,
    apply_pruning,
)
from .param_schema import (
    infer_layer_type,
    build_type_map,
    get_prunable_types,
    register_type_rule,
    list_registered_families,
)

__all__ = [
    'ImportanceScorer', 'MagnitudeScorer', 'FirstOrderScorer',
    'ResidualMagnitudeScorer',
    'get_importance_scorer', 'list_importance_methods',
    'AllocationStrategy', 'UniformAllocation', 'GammaAdaptiveAllocation', 'WeibullAdaptiveAllocation',
    'get_allocation_strategy', 'list_allocation_strategies',
    'Pruner', 'filter_prunable_params', 'apply_pruning',
    'combine_scores_2d_with_protection', 'apply_magnitude_protection',
    'infer_layer_type', 'build_type_map', 'get_prunable_types',
    'register_type_rule', 'list_registered_families',
]
