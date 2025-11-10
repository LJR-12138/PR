"""生成离线增强的检测训练数据集。"""

from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.utils.config import load_config
from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass
class Args:
    """命令行参数。"""

    config: Path = Path("configs/local.yaml")
    source_split: str = "train"
    dest_split: str = "train_aug"
    variants: int = 2
    clean: bool = True
    seed: Optional[int] = None


def _copy_label(label_src: Path, label_dst: Path) -> None:
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label_src, label_dst)


def _augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    """对图像执行一组仅影响像素的增强操作。"""

    img = image.convert("RGB")

    if rng.random() < 0.8:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.85, 1.25))
    if rng.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.35))
    if rng.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.3))
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.8)))
    if rng.random() < 0.5:
        cutoff = rng.uniform(0.0, 3.0)
        img = ImageOps.autocontrast(img, cutoff=cutoff)
    if rng.random() < 0.6:
        arr = np.asarray(img).astype(np.float32)
        local_rng = np.random.default_rng(rng.randint(0, 1_000_000_000))
        noise = local_rng.normal(loc=0.0, scale=rng.uniform(4.0, 12.0), size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    repo_root = resolve_path(cfg.get("paths.repo_root", "."), cfg.source.parent)
    dataset_root = resolve_path(cfg.get("paths.labelled_dataset_dir", "data/furseal/datasets"), repo_root)

    image_subdir = cfg.get("paths.detection_image_subdir", "images")
    label_subdir = cfg.get("paths.detection_label_subdir", "labels")

    source_base = dataset_root / args.source_split
    dest_base = dataset_root / args.dest_split

    source_img_dir = source_base / image_subdir
    source_lbl_dir = source_base / label_subdir

    dest_img_dir = dest_base / image_subdir
    dest_lbl_dir = dest_base / label_subdir

    if not source_img_dir.exists():
        raise FileNotFoundError(f"源图像目录不存在: {source_img_dir}")
    if not source_lbl_dir.exists():
        raise FileNotFoundError(f"源标注目录不存在: {source_lbl_dir}")

    if args.clean and dest_base.exists():
        LOGGER.info("删除目的目录 %s", dest_base)
        shutil.rmtree(dest_base)

    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed or cfg.get("project.seed", 1337))

    image_paths = sorted(list(source_img_dir.glob("*.jpg")) + list(source_img_dir.glob("*.JPG")) + list(source_img_dir.glob("*.png")))
    if not image_paths:
        LOGGER.warning("源目录 %s 中未找到图像", source_img_dir)
        return

    copied = 0
    augmented = 0

    for image_path in image_paths:
        label_path = source_lbl_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            LOGGER.warning("未找到与 %s 对应的标注文件，跳过", image_path.name)
            continue

        dest_image_path = dest_img_dir / image_path.name
        shutil.copy2(image_path, dest_image_path)
        _copy_label(label_path, dest_lbl_dir / label_path.name)
        copied += 1

        with Image.open(image_path) as img:
            for variant_idx in range(args.variants):
                augmented_img = _augment_image(img, rng)
                aug_name = f"{image_path.stem}_aug{variant_idx + 1:02d}{image_path.suffix}"
                aug_path = dest_img_dir / aug_name
                augmented_img.save(aug_path, quality=95)
                _copy_label(label_path, dest_lbl_dir / f"{image_path.stem}_aug{variant_idx + 1:02d}.txt")
                augmented += 1

    LOGGER.info("完成复制 %d 张原始图像，额外生成 %d 张增强图像", copied, augmented)
    LOGGER.info("新的训练分割位于 %s", dest_base)


if __name__ == "__main__":
    main(tyro.cli(Args))
