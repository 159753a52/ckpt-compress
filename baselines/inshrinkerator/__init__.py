"""Inshrinkerator compression method.

Modules:
  - inshrinkerator.py: core Inshrinkerator pipeline
  - per_type_search.py: per-layer-type pruning ratio search (two-phase grid search)
  - per_type_allocation.py: AllocationStrategy subclass using pre-searched ratios
  - approx_kmeans.py / sketch.py: DDSketch-based approximate K-means quantization
  - delta_encoding.py: delta encoding for checkpoint compression
  - partition.py: parameter partitioning utilities
  - metrics.py: compression metrics
"""

from .per_type_search import (
    SearchConfig,
    SearchResult,
    apply_pruning_per_type,
    estimate_global_ratio,
    search_best_config,
    save_search_result,
    load_search_result,
)
from .per_type_allocation import PerTypeAllocation
