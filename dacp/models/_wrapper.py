"""Shared behavior for model wrappers that expose an inner model directly."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


class DelegatingModel(nn.Module):
    """Keep wrapper state and parameter names compatible with the inner model."""

    model: nn.Module

    @property
    def _wrapped_model(self) -> nn.Module:
        return self.model

    def parameters(self, recurse: bool = True):
        return self._wrapped_model.parameters(recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self._wrapped_model.named_parameters(prefix, recurse)

    def state_dict(self, *args: Any, **kwargs: Any):
        return self._wrapped_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self._wrapped_model.load_state_dict(state_dict, strict)

    def train(self, mode: bool = True):
        self._wrapped_model.train(mode)
        return self

    def eval(self):
        self._wrapped_model.eval()
        return self


__all__ = ["DelegatingModel"]
