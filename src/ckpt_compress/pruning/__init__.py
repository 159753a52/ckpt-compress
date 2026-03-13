"""统一剪枝框架。

提供统一的重要性得分计算、剪枝率分配和剪枝执行接口。
支持多种方法的公平对比。

用法:
    from ckpt_compress.pruning import Pruner
    
    pruner = Pruner(importance='second-order', allocation='gamma-adaptive', alpha=0.5)
    scores = pruner.compute_scores(weights, gradients, exp_avg_sq)
    layer_ratios = pruner.compute_layer_ratios(scores, global_ratio=0.5)
    model, masks, actual_ratio = pruner.prune(model, scores, layer_ratios, device)
"""

from .importance import (
    ImportanceScorer,
    MagnitudeScorer,
    FirstOrderScorer,
    SecondOrderScorer,
    ResidualMagnitudeScorer,
    get_importance_scorer,
    list_importance_methods,
    IMPORTANCE_REGISTRY,
)
from .allocation import (
    AllocationStrategy,
    UniformAllocation,
    GammaAdaptiveAllocation,
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
from .per_type_allocation import PerTypeAllocation
from .inshrinkerator_search import (
    SearchConfig,
    SearchResult,
    apply_pruning_per_type,
    estimate_global_ratio,
    search_best_config,
    save_search_result,
    load_search_result,
)

__all__ = [
    'ImportanceScorer', 'MagnitudeScorer', 'FirstOrderScorer',
    'SecondOrderScorer', 'ResidualMagnitudeScorer',
    'get_importance_scorer', 'list_importance_methods',
    'AllocationStrategy', 'UniformAllocation', 'GammaAdaptiveAllocation',
    'PerTypeAllocation',
    'get_allocation_strategy', 'list_allocation_strategies',
    'Pruner', 'filter_prunable_params', 'apply_pruning',
    'infer_layer_type', 'build_type_map', 'get_prunable_types',
    'register_type_rule', 'list_registered_families',
    'SearchConfig', 'SearchResult',
    'apply_pruning_per_type', 'estimate_global_ratio',
    'search_best_config', 'save_search_result', 'load_search_result',
]
