"""CLI wrapper to fine-tune YOLO detector."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import tyro

from src.detection.yolo_trainer import train_detector
from src.utils.config import load_config
from src.utils.logging import init_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class Args:
    config: Path = Path("configs/local.yaml")


def main(args: Args) -> None:
    cfg = load_config(args.config)
    init_logging(cfg.get("logging.level", "INFO"))
    LOGGER.info("Starting detector training")
    best_ckpt = train_detector(cfg)
    LOGGER.info("Training finished. Best checkpoint: %s", best_ckpt)


if __name__ == "__main__":
    main(tyro.cli(Args))
