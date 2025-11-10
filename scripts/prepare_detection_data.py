"""Utility helpers for preparing YOLO training data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import tyro

from src.data.datasets import build_detection_catalog
from src.utils.config import load_config
from src.utils.logging import init_logging
from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass
class Args:
    config: Path = Path("configs/local.yaml")
    export_manifest: Optional[Path] = None


def main(args: Args) -> None:
    cfg = load_config(args.config)
    init_logging(cfg.get("logging.level", "INFO"))

    repo_root = resolve_path(cfg.get("paths.repo_root", "."), cfg.source.parent)
    catalog = build_detection_catalog(cfg.get("paths", {}), repo_root)

    summary_rows = []
    for split, dataset in catalog.iter_available():
        dataset.validate()
        images = list(dataset.images_dir.glob("*.jpg")) + list(dataset.images_dir.glob("*.png"))
        labels = list(dataset.labels_dir.glob("*.txt"))
        LOGGER.info(
            "%s split: %s images, %s label files", split, len(images), len(labels)
        )
        summary_rows.append(
            {
                "split": split,
                "images": len(images),
                "labels": len(labels),
                "images_dir": dataset.images_dir.as_posix(),
                "labels_dir": dataset.labels_dir.as_posix(),
            }
        )

    if args.export_manifest:
        manifest_path = args.export_manifest
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(manifest_path, index=False)
        LOGGER.info("Exported manifest to %s", manifest_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
