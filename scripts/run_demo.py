"""一键串联整个皮毛海豹识别演示流程。"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from scripts.analyze_behaviour import Args as BehaviourArgs, main as behaviour_main
from scripts.build_embeddings import Args as EmbeddingArgs, main as embeddings_main
from scripts.cluster_identities import Args as ClusterArgs, main as cluster_main
from scripts.run_detection import Args as DetectionArgs, main as detection_main
from scripts.train_detector import Args as TrainArgs, main as train_main
from src.utils.config import load_config
from src.utils.logging import init_logging
from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass
class DemoArgs:
    config: Path = Path("configs/local.yaml")
    split: str = "valid"
    retrain: bool = False
    limit: Optional[int] = None
    clean_outputs: bool = True


def main(args: DemoArgs) -> None:
    cfg = load_config(args.config)
    init_logging(cfg.get("logging.level", "INFO"))
    repo_root = resolve_path(cfg.get("paths.repo_root", "."), cfg.source.parent)

    if args.clean_outputs:
        _clean_outputs(cfg, repo_root)

    weights_key = cfg.get(
        "training.detector.best_weights",
        "outputs/training/detector/exp/weights/best.pt",
    )
    weights_path = resolve_path(Path(weights_key), repo_root)

    if args.retrain or not weights_path.exists():
        LOGGER.info("开始重新训练检测器...")
        train_main(TrainArgs(config=args.config))
        cfg = load_config(args.config)
        weights_key = cfg.get("training.detector.best_weights", weights_key)
        weights_path = resolve_path(Path(weights_key), repo_root)

    if not weights_path.exists():
        raise FileNotFoundError(f"检测器权重不存在: {weights_path}")

    LOGGER.info("使用权重 %s 进行检测", weights_path)
    detection_main(DetectionArgs(config=args.config, split=args.split, weights=None))

    LOGGER.info("开始提取人脸补丁与嵌入...")
    embeddings_main(EmbeddingArgs(config=args.config, weights=None, limit=args.limit))

    LOGGER.info("聚类身份并生成行为报告...")
    cluster_main(ClusterArgs(config=args.config))
    behaviour_main(BehaviourArgs(config=args.config))

    LOGGER.info("演示流程执行完毕。输出位于 outputs/ 目录。")


def _clean_outputs(cfg, repo_root: Path) -> None:
    paths_cfg = cfg.get("paths", {})

    detection_dir = resolve_path(paths_cfg.get("inference_output_dir", "outputs/detections"), repo_root)
    faces_dir = resolve_path(paths_cfg.get("face_patch_dir", "outputs/faces"), repo_root)
    embeddings_path = resolve_path(paths_cfg.get("embeddings_path", "outputs/embeddings/embeddings.parquet"), repo_root)
    embeddings_dir = embeddings_path.parent
    reports_dir = resolve_path(paths_cfg.get("behaviour_report_dir", "outputs/reports"), repo_root)

    for target in [detection_dir, faces_dir, embeddings_dir, reports_dir]:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main(tyro.cli(DemoArgs))
