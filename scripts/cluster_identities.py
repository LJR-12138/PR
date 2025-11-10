"""Cluster fur-seal face embeddings into individual identities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro

from src.recognition.cluster import IdentityAssignment, cluster_embeddings
from src.recognition.storage import load_embeddings
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
    repo_root = resolve_path(cfg.get("paths.repo_root", "."), cfg.source.parent)

    embeddings_path = resolve_path(cfg.get("paths.embeddings_path", "outputs/embeddings/embeddings.parquet"), repo_root)
    embeddings = load_embeddings(embeddings_path)
    LOGGER.info("Loaded %s embeddings", len(embeddings))

    clustering_cfg = cfg.get("clustering", {})
    assignments = cluster_embeddings(
        embeddings,
        eps=clustering_cfg.get("eps", 0.5),
        min_samples=clustering_cfg.get("min_samples", 3),
        metric=clustering_cfg.get("metric", "cosine"),
        use_faiss=clustering_cfg.get("use_faiss", True),
        faiss_nlist=clustering_cfg.get("faiss_nlist", 100),
        faiss_nprobe=clustering_cfg.get("faiss_nprobe", 10),
    )

    save_path = resolve_path(cfg.get("paths.embeddings_path", "outputs/embeddings/embeddings.parquet"), repo_root)
    identities_path = save_path.with_name("identities.parquet")
    _save_assignments(assignments, identities_path)
    LOGGER.info("Identity assignments written to %s", identities_path)


def _save_assignments(assignments: list[IdentityAssignment], path: Path) -> None:
    records = []
    for assignment in assignments:
        face = assignment.embedding.face
        records.append(
            {
                "identity": assignment.identity,
                "image_path": face.image_path.as_posix(),
                "patch_path": face.patch_path.as_posix(),
                "score": face.score,
                "bbox_x1": face.bbox_xyxy[0],
                "bbox_y1": face.bbox_xyxy[1],
                "bbox_x2": face.bbox_xyxy[2],
                "bbox_y2": face.bbox_xyxy[3],
            }
        )
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


if __name__ == "__main__":
    main(tyro.cli(Args))
