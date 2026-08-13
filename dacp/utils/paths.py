"""Portable repository data and model path resolution."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_ENV = "DACP_DATA_ROOT"
MODEL_ROOT_ENV = "DACP_MODEL_ROOT"


def _resolve_root(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def data_root(override: str | Path | None = None) -> Path:
    """Return the configured data root, whether or not it exists yet."""
    value = override if override is not None else os.environ.get(DATA_ROOT_ENV)
    return _resolve_root(value, PROJECT_ROOT / "data")


def model_root(override: str | Path | None = None) -> Path:
    """Return the model root, defaulting beneath the configured data root."""
    value = override if override is not None else os.environ.get(MODEL_ROOT_ENV)
    return _resolve_root(value, data_root() / "models")


def resolve_model_source(
    local_name: str,
    remote_name: str,
    *,
    root: str | Path | None = None,
) -> str:
    """Prefer a configured local snapshot, otherwise return the remote model id."""
    candidate = model_root(root) / local_name
    return str(candidate) if candidate.is_dir() else remote_name


def resolve_data_file(
    relative_path: str | Path,
    *,
    root: str | Path | None = None,
) -> Path:
    """Resolve one repository-relative dataset file under a configurable root."""
    return (data_root(root) / relative_path).resolve()


__all__ = [
    "DATA_ROOT_ENV",
    "MODEL_ROOT_ENV",
    "PROJECT_ROOT",
    "data_root",
    "model_root",
    "resolve_data_file",
    "resolve_model_source",
]
