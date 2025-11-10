"""Extract face crops and compute embeddings for fur-seal imagery."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from src.recognition.embedding_model import FaceEmbedder
from src.recognition.face_extractor import FaceExtractor
from src.recognition.storage import save_embeddings
from src.utils.config import load_config
from src.utils.logging import init_logging
from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass
class Args:
    config: Path = Path("configs/local.yaml")
    weights: Optional[Path] = None
    limit: Optional[int] = None


def main(args: Args) -> None:
    cfg = load_config(args.config)
    init_logging(cfg.get("logging.level", "INFO"))
    repo_root = resolve_path(cfg.get("paths.repo_root", "."), cfg.source.parent)

    weights_path = args.weights or Path(
        cfg.get("training.detector.best_weights", "outputs/training/detector/exp/weights/best.pt")
    )
    weights_path = resolve_path(weights_path, repo_root)

    image_root = resolve_path(cfg.get("paths.raw_dataset_dir", "data/furseal/task_datasets"), repo_root)
    if not image_root.exists():
        raise FileNotFoundError(f"Raw dataset directory missing: {image_root}")

    image_paths = sorted(
        path
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for path in image_root.rglob(ext)
    )
    if args.limit:
        image_paths = image_paths[: args.limit]

    LOGGER.info("Extracting faces from %s images", len(image_paths))
    patch_dir = resolve_path(cfg.get("paths.face_patch_dir", "outputs/faces"), repo_root)

    extractor = FaceExtractor(
        model_weights=weights_path,
        output_dir=patch_dir,
        device=cfg.get("project.device", "cuda"),
        conf=cfg.get("inference.confidence_threshold", 0.3),
        iou=cfg.get("inference.iou_threshold", 0.5),
        target_size=cfg.get("preprocessing.target_size", 160) or 160,
        min_face_size=cfg.get("preprocessing.min_face_size", 48) or 48,
        quality_threshold=cfg.get("preprocessing.quality_threshold", 0.6) or 0.6,
        duplicate_iou_threshold=cfg.get("preprocessing.duplicate_iou_threshold", 0.9) or 0.9,
    )
    patches = extractor.extract(image_paths)
    LOGGER.info("Extracted %s face patches", len(patches))

    recognition_cfg = cfg.get("training.recognition", {})
    embedder = FaceEmbedder(
        backbone=recognition_cfg.get("backbone", "facenet-inception"),
        device=cfg.get("project.device", "cuda"),
        batch_size=recognition_cfg.get("batch_size", 64),
        num_workers=recognition_cfg.get("num_workers", 0),
    )
    embeddings = embedder.encode(patches)

    embeddings_path = resolve_path(cfg.get("paths.embeddings_path", "outputs/embeddings/embeddings.parquet"), repo_root)
    save_embeddings(embeddings, embeddings_path)
    LOGGER.info("Saved embeddings to %s", embeddings_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
