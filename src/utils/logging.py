"""Logging configuration helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def init_logging(level: str = "INFO", log_dir: Optional[Path] = None) -> None:
    """Initialise the application wide logger with Rich formatting."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler]
    handlers = [RichHandler(console=Console(), rich_tracebacks=True, markup=True)]
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=handlers,
    )
