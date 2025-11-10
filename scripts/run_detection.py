"""Run inference with a trained detector on a dataset split."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from src.data.datasets import build_detection_catalog
from src.detection.predictor import run_detection
from src.utils.config import load_config
from src.utils.logging import init_logging
from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass
class Args:
    config: Path = Path("configs/local.yaml")
    split: str = "val"
    weights: Optional[Path] = None


def main(args: Args) -> None:
    cfg = load_config(args.config)
    init_logging(cfg.get("logging.level", "INFO"))
    repo_root = resolve_path(cfg.get("paths.repo_root", "."), cfg.source.parent)

    detector_weights = args.weights or Path(
        cfg.get("training.detector.best_weights", "outputs/training/detector/exp/weights/best.pt")
    )
    detector_weights = resolve_path(detector_weights, repo_root)

    paths_cfg = cfg.get("paths", {})
    catalog = build_detection_catalog(paths_cfg, repo_root)

    split_aliases = {
        "train": "train",
        "val": "val",
        "test": "test",
        paths_cfg.get("detection_train_split", "train"): "train",
        paths_cfg.get("detection_val_split", "val"): "val",
    }

    test_split_name = paths_cfg.get("detection_test_split")
    if test_split_name:
        split_aliases[test_split_name] = "test"

    canonical_split = split_aliases.get(args.split, args.split)
    dataset = getattr(catalog, canonical_split, None)
    if dataset is None:
        known = ", ".join(sorted({alias for alias in split_aliases if split_aliases[alias] in {"train", "val", "test"}}))
        raise ValueError(f"Unknown split '{args.split}'. Expected one of: {known}")

    image_paths = sorted(dataset.images_dir.glob("*.jpg")) + sorted(dataset.images_dir.glob("*.png"))
    if not image_paths:
        LOGGER.warning("No images found in %s", dataset.images_dir)
        return

    output_dir = resolve_path(cfg.get("paths.inference_output_dir", "outputs/detections"), repo_root)
    run_detection(
        model_weights=detector_weights,
        image_paths=image_paths,
        output_dir=output_dir,
        conf=cfg.get("inference.confidence_threshold", 0.3),
        iou=cfg.get("inference.iou_threshold", 0.5),
        imgsz=cfg.get("inference.img_size"),
        augment=cfg.get("inference.augment_tta", False),
        post_nms_duplicate_iou=cfg.get("inference.post_nms_duplicate_iou"),
        device=cfg.get("project.device", "cuda"),
        max_det=cfg.get("inference.max_det", 100),
        save_txt=cfg.get("inference.save_txt", True),
        save_conf=cfg.get("inference.save_conf", True),
    )
    LOGGER.info("Detection outputs written to %s", output_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
