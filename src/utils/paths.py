"""Helpers for resolving project paths consistently."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_project_root(start: Optional[str | Path] = None) -> Path:
    """Locate the project root by walking upwards until pyproject.toml is found."""
    current = Path(start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to cwd when pyproject is missing
    return current


def resolve_path(path_str: str | Path, base: Optional[Path] = None) -> Path:
    """Convert a relative path into an absolute Path relative to base or project root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    root = base or find_project_root()
    return (root / path).resolve()
