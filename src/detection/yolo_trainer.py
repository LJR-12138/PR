"""Wrapping Ultralytics YOLO training for fur-seal face detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

from src.data.datasets import build_detection_catalog
from src.utils.config import Config
from src.utils.experiments import log_detector_run
from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


def train_detector(cfg: Config) -> Path:
    """Fine-tune a YOLO model on the fur-seal dataset.

    Returns the path to the best checkpoint.
    """
    training_cfg: Dict[str, Dict[str, str]] = cfg.get("training", {})
    detector_cfg: Dict[str, str] = training_cfg.get("detector", {})
    path_cfg: Dict[str, str] = cfg.get("paths", {})

    repo_root = resolve_path(cfg.get("paths.repo_root", "."), cfg.source.parent)
    catalog = build_detection_catalog(path_cfg, repo_root)
    dataset_yaml = _write_dataset_yaml(catalog, repo_root)

    model_name = detector_cfg.get("model", "yolov8n.pt")
    epochs = int(detector_cfg.get("epochs", 50))
    batch = int(detector_cfg.get("batch", 16))
    img = int(detector_cfg.get("img_size", 640))
    project_dir = resolve_path(
        detector_cfg.get("project", "outputs/training/detector"),
        repo_root,
    )
    project_dir.mkdir(parents=True, exist_ok=True)
    run_name = detector_cfg.get("run_name", "exp")

    LOGGER.info("Loading YOLO model '%s'", model_name)
    model = YOLO(model_name)

    LOGGER.info(
        "Starting training for %s epochs (batch=%s, img=%s) with dataset %s",
        epochs,
        batch,
        img,
        dataset_yaml,
    )

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=img,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
        lr0=detector_cfg.get("lr0", 0.01),
        lrf=detector_cfg.get("lrf", 0.01),
        warmup_epochs=detector_cfg.get("warmup_epochs", 3.0),
        patience=detector_cfg.get("patience", 20),
        augment=detector_cfg.get("augment", True),
        mosaic=detector_cfg.get("mosaic", 1.0),
        mixup=detector_cfg.get("mixup", 0.0),
        copy_paste=detector_cfg.get("copy_paste", 0.0),
        dropout=detector_cfg.get("dropout", 0.0),
        workers=detector_cfg.get("workers", 4),
        optimizer=detector_cfg.get("optimizer", "AdamW"),
        val=detector_cfg.get("val", True),
        device=cfg.get("project.device", "cuda"),
        **_extra_train_kwargs(detector_cfg),
    )

    best_ckpt: Path
    if hasattr(model, "trainer") and getattr(model.trainer, "best", None):
        best_ckpt = Path(model.trainer.best)
    elif hasattr(results, "trainer") and getattr(results.trainer, "best", None):
        best_ckpt = Path(results.trainer.best)
    else:
        best_ckpt = project_dir / run_name / "weights" / "best.pt"
    LOGGER.info("Best checkpoint saved at %s", best_ckpt)

    try:
        log_detector_run(cfg, project_dir / run_name, best_ckpt)
    except Exception as exc:  # pragma: no cover - do not fail training on logging errors
        LOGGER.warning("Failed to log detector experiment: %s", exc)
    return best_ckpt


def _write_dataset_yaml(catalog, base_dir: Path) -> Path:
    """Emit a temporary Ultralytics dataset YAML file."""
    import yaml

    dataset_dir = resolve_path("outputs/training/tmp", base_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = dataset_dir / "furseal_detection.yaml"

    data_dict = {
        "path": "",
        "train": str(catalog.train.images_dir),
        "val": str(catalog.val.images_dir),
        "names": ["fur_seal_face"],
    }

    if catalog.test is not None:
        data_dict["test"] = str(catalog.test.images_dir)

    yaml_path.write_text(yaml.safe_dump(data_dict), encoding="utf-8")
    return yaml_path


def _extra_train_kwargs(detector_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract optional Ultralytics training arguments from the config."""
    allowed_keys = {
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "erasing",
        "auto_augment",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "hsv",
        "rect",
        "pretrained",
    }
    extras: Dict[str, Any] = {}
    for key in allowed_keys:
        if key in detector_cfg:
            extras[key] = detector_cfg[key]
    return extras
