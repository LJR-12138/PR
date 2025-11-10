"""Serialization helpers for face patches and embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import torch

from src.recognition.embedding_model import FaceEmbedding
from src.recognition.face_extractor import FacePatch


def save_embeddings(embeddings: Sequence[FaceEmbedding], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for item in embeddings:
        face = item.face
        records.append(
            {
                "image_path": face.image_path.as_posix(),
                "patch_path": face.patch_path.as_posix(),
                "bbox_x1": face.bbox_xyxy[0],
                "bbox_y1": face.bbox_xyxy[1],
                "bbox_x2": face.bbox_xyxy[2],
                "bbox_y2": face.bbox_xyxy[3],
                "score": face.score,
                "embedding": item.embedding.cpu().numpy().astype("float32").tolist(),
            }
        )
    df = pd.DataFrame(records)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_json(path, orient="records", indent=2)


def load_embeddings(path: str | Path) -> List[FaceEmbedding]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_json(path)

    embeddings: List[FaceEmbedding] = []
    for row in df.itertuples(index=False):
        face = FacePatch(
            image_path=Path(row.image_path),
            patch_path=Path(row.patch_path),
            bbox_xyxy=(float(row.bbox_x1), float(row.bbox_y1), float(row.bbox_x2), float(row.bbox_y2)),
            score=float(row.score),
        )
        vector = torch.tensor(row.embedding, dtype=torch.float32)
        embeddings.append(FaceEmbedding(face=face, embedding=vector))
    return embeddings
