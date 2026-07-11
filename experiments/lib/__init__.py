"""
实验库模块

提供实验相关的核心功能：
- models: 模型加载
- data: 数据加载
- evaluation: 评估
- results: 结果管理
- importance_compare: 重要性得分批量计算
"""

from .models import load_model, get_model_type
from .data import get_data_loaders, cache_batches
from .evaluation import evaluate, compute_quality_drop

# results 依赖 pandas，延迟导入以减少基础依赖
def __getattr__(name):
    if name in ('ResultManager', 'save_results', 'print_results_table'):
        from .results import ResultManager, save_results, print_results_table
        globals().update({
            'ResultManager': ResultManager,
            'save_results': save_results,
            'print_results_table': print_results_table,
        })
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'load_model', 'get_model_type',
    'get_data_loaders', 'cache_batches',
    'evaluate', 'compute_quality_drop',
    'ResultManager', 'save_results', 'print_results_table',
]
