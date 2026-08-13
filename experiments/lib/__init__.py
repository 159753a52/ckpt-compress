"""Shared experiment APIs, imported lazily to keep utility startup light."""

from importlib import import_module

_EXPORTS = {
    "load_model": (".models", "load_model"),
    "get_model_type": (".models", "get_model_type"),
    "get_data_loaders": (".data", "get_data_loaders"),
    "cache_batches": (".data", "cache_batches"),
    "evaluate": (".evaluation", "evaluate"),
    "compute_quality_drop": (".evaluation", "compute_quality_drop"),
    "ResultManager": (".results", "ResultManager"),
    "save_results": (".results", "save_results"),
    "print_results_table": (".results", "print_results_table"),
}


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "load_model",
    "get_model_type",
    "get_data_loaders",
    "cache_batches",
    "evaluate",
    "compute_quality_drop",
    "ResultManager",
    "save_results",
    "print_results_table",
]
