"""Quick diagnostics for environment and dataset readiness."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from src.data.datasets import build_detection_catalog
from src.utils.config import load_config
from src.utils.logging import init_logging
from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass
class Args:
    config: Path = Path("configs/local.yaml")


def main(args: Args) -> None:
    cfg = load_config(args.config)
    init_logging(cfg.get("logging.level", "INFO"))

    LOGGER.info("Using config at %s", cfg.source)
    device = cfg.get("project.device", "cuda")
    if device == "cuda":
        LOGGER.info("CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            LOGGER.info("CUDA device count: %s", torch.cuda.device_count())
    else:
        LOGGER.info("Running on device %s", device)

    path_cfg = cfg.get("paths", {})
    repo_root = resolve_path(cfg.get("paths.repo_root", "."), cfg.source.parent)
    catalog = build_detection_catalog(path_cfg, repo_root)
    for split, dataset in catalog.iter_available():
        LOGGER.info("Validating %s split", split)
        dataset.validate()

    raw_dataset = resolve_path(path_cfg.get("raw_dataset_dir", "data/furseal/task_datasets"), repo_root)
    if raw_dataset.exists():
        LOGGER.info("Raw dataset folder contains %s files", sum(1 for _ in raw_dataset.rglob("*")))
    else:
        LOGGER.warning("Raw dataset directory missing: %s", raw_dataset)

    LOGGER.info("Setup check complete")


if __name__ == "__main__":
    main(tyro.cli(Args))
