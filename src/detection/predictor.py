"""Utilities to run batched inference with a trained YOLO detector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from ultralytics import YOLO

from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


def run_detection(
    model_weights: str | Path,
    image_paths: Iterable[str | Path],
    output_dir: str | Path,
    conf: float = 0.25,
    iou: float = 0.5,
    device: str = "cuda",
    imgsz: int | tuple[int, int] | None = None,
    augment: bool = False,
    post_nms_duplicate_iou: float | None = None,
    max_det: int = 100,
    save_txt: bool = True,
    save_conf: bool = True,
) -> List[Path]:
    """Run inference and return the list of generated prediction files."""
    weights_path = resolve_path(model_weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [Path(path) for path in image_paths]
    LOGGER.info("Running detection on %s images", len(image_paths))
    # Persist outputs under the configured directory instead of the Ultralytics default.
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=conf,
        iou=iou,
        device=device,
        imgsz=imgsz,
        augment=augment,
        save_txt=save_txt,
        save_conf=save_conf,
        save=True,
        project=str(out_dir),
        name="latest",
        exist_ok=True,
        max_det=max_det,
        verbose=True,
    )

    generated: List[Path] = []
    for res in results:
        if res.save_dir:
            generated.append(Path(res.save_dir))

    if save_txt:
        labels_dir = out_dir / "latest" / "labels"
        _deduplicate_label_files(labels_dir, iou_thr=post_nms_duplicate_iou)
    return generated


def _deduplicate_label_files(labels_dir: Path, iou_thr: float | None = None) -> None:
    if not labels_dir.exists():
        return

    for label_path in labels_dir.glob("*.txt"):
        raw = label_path.read_text().strip().splitlines()
        if not raw:
            continue
        # First pass: drop exact duplicates (string-equal after rounding)
        seen: Set[Tuple[str, ...]] = set()
        lines: List[List[float]] = []
        for line in raw:
            parts = line.strip().split()
            if not parts:
                continue
            rounded = (parts[0],) + tuple(f"{float(x):.6f}" for x in parts[1:])
            if rounded in seen:
                continue
            seen.add(rounded)
            try:
                nums = list(map(float, parts))
                if len(nums) < 6:
                    continue
                lines.append(nums)
            except Exception:  # pragma: no cover
                continue

        if iou_thr is not None and iou_thr > 0:
            # Second pass: IoU-based suppression. Keep highest-confidence boxes.
            # Format per line: class cx cy w h conf (all normalised 0..1)
            boxes = []
            for vals in lines:
                cls, cx, cy, w, h, conf = vals[:6]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                boxes.append([cls, x1, y1, x2, y2, conf])
            # Sort by confidence desc
            boxes.sort(key=lambda b: b[5], reverse=True)
            kept: List[List[float]] = []
            for b in boxes:
                suppress = False
                for k in kept:
                    if _iou_xyxy(b[1:5], k[1:5]) >= iou_thr:
                        suppress = True
                        break
                if not suppress:
                    kept.append(b)
            out_lines = [f"{int(b[0])} {(b[1]+b[3])/2:.6f} {(b[2]+b[4])/2:.6f} {(b[3]-b[1]):.6f} {(b[4]-b[2]):.6f} {b[5]:.6f}" for b in kept]
        else:
            out_lines = [" ".join(str(x) for x in vals) for vals in lines]

        label_path.write_text("\n".join(out_lines) + "\n")


def _iou_xyxy(a: List[float] | Tuple[float, float, float, float], b: List[float] | Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
