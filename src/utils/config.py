"""Configuration loading utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(slots=True)
class Config:
    """Lightweight wrapper around a nested configuration dictionary."""

    data: Dict[str, Any]
    source: Path

    def get(self, dotted_key: str, default: Any | None = None) -> Any:
        """Retrieve a value using dotted notation (e.g. "training.detector.epochs")."""
        parts = dotted_key.split(".")
        node: Any = self.data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def as_dict(self) -> Dict[str, Any]:
        return self.data


def load_config(config_path: str | os.PathLike[str]) -> Config:
    """Load a YAML config file, expanding environment variables along the way."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    expanded = _expand_env(raw)
    return Config(data=expanded, source=path)


def _expand_env(obj: Any) -> Any:
    """Recursively expand environment variables within strings."""
    if isinstance(obj, dict):
        return {key: _expand_env(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(item) for item in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj
