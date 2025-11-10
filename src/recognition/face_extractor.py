"""Crop and store fur-seal face patches from YOLO detections."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FacePatch:
    """Metadata for an extracted face patch."""

    image_path: Path
    patch_path: Path
    bbox_xyxy: tuple[float, float, float, float]
    score: float


class FaceExtractor:
    """Extract high quality fur-seal face crops using a trained detector."""

    def __init__(
        self,
        model_weights: str | Path,
        output_dir: str | Path,
        device: str = "cuda",
        conf: float = 0.3,
        iou: float = 0.5,
        target_size: int = 160,
        min_face_size: int = 48,
        quality_threshold: float = 0.6,
        duplicate_iou_threshold: float = 0.9,
    ) -> None:
        self.model = YOLO(str(resolve_path(model_weights)))
        self.output_dir = resolve_path(output_dir)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.target_size = target_size
        self.min_face_size = min_face_size
        self.quality_threshold = quality_threshold
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, image_paths: Iterable[str | Path]) -> List[FacePatch]:
        """Run extraction for the provided images."""
        patches: List[FacePatch] = []
        for image_path in image_paths:
            image_path = resolve_path(image_path)
            LOGGER.debug("Processing image %s", image_path)
            results = self.model.predict(
                source=str(image_path),
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            if not results:
                continue
            patches.extend(self._handle_single_image(image_path, results[0]))
        return patches

    def _handle_single_image(self, image_path: Path, result) -> Sequence[FacePatch]:
        face_patches: List[FacePatch] = []
        kept_boxes: List[tuple[float, float, float, float]] = []
        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size

            for idx, box in enumerate(result.boxes):
                xyxy = box.xyxy.cpu().numpy().astype(np.float32)[0]
                confidence = float(box.conf.cpu().numpy()[0])
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1

                if min(w, h) < self.min_face_size:
                    LOGGER.debug("Skipping tiny detection %.1fx%.1f", w, h)
                    continue
                if confidence < self.quality_threshold:
                    continue
                current_box = (float(x1), float(y1), float(x2), float(y2))
                if any(self._bbox_iou(current_box, prev) >= self.duplicate_iou_threshold for prev in kept_boxes):
                    LOGGER.debug("Skipping duplicate detection with IoU >= %.2f", self.duplicate_iou_threshold)
                    continue

                x1_clip = max(0, int(x1))
                y1_clip = max(0, int(y1))
                x2_clip = min(width, int(x2))
                y2_clip = min(height, int(y2))
                crop = image.crop((x1_clip, y1_clip, x2_clip, y2_clip))
                crop = crop.resize((self.target_size, self.target_size))

                patch_path = self.output_dir / f"{image_path.stem}_face_{idx:03d}.png"
                crop.save(patch_path)
                kept_boxes.append(current_box)
                face_patches.append(
                    FacePatch(
                        image_path=image_path,
                        patch_path=patch_path,
                        bbox_xyxy=current_box,
                        score=confidence,
                    )
                )
        return face_patches

    @staticmethod
    def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter == 0.0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)
