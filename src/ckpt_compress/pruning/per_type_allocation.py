"""Per-layer-type allocation strategy (Inshrinkerator-like).

Reads pre-searched per-type pruning ratios from a JSON file,
then maps each parameter to its type and assigns the corresponding ratio.
Supports proportional scaling to arbitrary global target ratios.
"""

import json
from typing import Dict, Optional

import torch

from .allocation import AllocationStrategy, ALLOCATION_REGISTRY
from .param_schema import infer_layer_type


class PerTypeAllocation(AllocationStrategy):
    """Allocation based on pre-searched per-layer-type ratios.

    Usage:
        alloc = PerTypeAllocation(
            per_type_ratios={'attn': 0.3, 'mlp': 0.4},
            source_global_ratio=0.35,
            model_family='gpt2',
        )
        layer_ratios = alloc.allocate(scores, global_prune_ratio=0.5)
    """

    def __init__(
        self,
        per_type_ratios: Optional[Dict[str, float]] = None,
        source_global_ratio: float = 0.0,
        model_family: str = 'gpt2',
        max_ratio: float = 0.95,
        json_path: Optional[str] = None,
        **kwargs,
    ):
        if json_path is not None:
            with open(json_path) as f:
                data = json.load(f)
            self._per_type_ratios = data['per_type_ratios']
            self._source_global_ratio = data['actual_global_ratio']
            self._best_metric = data.get('best_metric')
        else:
            if per_type_ratios is None:
                raise ValueError(
                    "Either per_type_ratios or json_path must be provided"
                )
            self._per_type_ratios = per_type_ratios
            self._source_global_ratio = source_global_ratio
            self._best_metric = None
        self._model_family = model_family
        self._max_ratio = max_ratio

    @property
    def name(self) -> str:
        return "per-type"

    @property
    def best_metric(self) -> Optional[str]:
        """The pruning metric selected by search (magnitude or sensitivity)."""
        return self._best_metric

    def get_importance_method(self) -> str:
        """Map search best_metric to importance method name for scoring."""
        mapping = {'magnitude': 'magnitude', 'sensitivity': 'first-order'}
        if self._best_metric and self._best_metric in mapping:
            return mapping[self._best_metric]
        return 'first-order'

    def allocate(
        self,
        scores: Dict[str, torch.Tensor],
        global_prune_ratio: float,
    ) -> Dict[str, float]:
        """Assign per-parameter pruning ratios based on layer type.

        If global_prune_ratio differs from the source ratio found during
        search, all per-type ratios are proportionally scaled.
        """
        if self._source_global_ratio > 0:
            scale = global_prune_ratio / self._source_global_ratio
        else:
            scale = 1.0

        result: Dict[str, float] = {}
        for param_name, score_tensor in scores.items():
            lt = infer_layer_type(param_name, score_tensor, self._model_family)
            if lt == 'skip' or lt not in self._per_type_ratios:
                result[param_name] = 0.0
            else:
                scaled = self._per_type_ratios[lt] * scale
                result[param_name] = min(scaled, self._max_ratio)
        return result

    @classmethod
    def from_json(cls, json_path: str, model_family: str = 'gpt2') -> 'PerTypeAllocation':
        """Load from a search result JSON file."""
        return cls(json_path=json_path, model_family=model_family)


ALLOCATION_REGISTRY["per-type"] = PerTypeAllocation
