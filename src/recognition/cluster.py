"""Identity clustering utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.cluster import DBSCAN

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency optional
    faiss = None

from src.recognition.embedding_model import FaceEmbedding

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IdentityAssignment:
    """Cluster label for an embedding."""

    embedding: FaceEmbedding
    identity: int


def cluster_embeddings(
    embeddings: Sequence[FaceEmbedding],
    eps: float = 0.5,
    min_samples: int = 3,
    metric: str = "cosine",
    use_faiss: bool = True,
    faiss_nlist: int = 100,
    faiss_nprobe: int = 10,
) -> List[IdentityAssignment]:
    """Cluster embeddings into individual identities."""
    if not embeddings:
        return []

    vectors = np.stack([emb.embedding.cpu().numpy() for emb in embeddings])
    vectors = _normalize(vectors)

    if use_faiss and faiss is not None and metric == "cosine":
        LOGGER.info("Clustering with FAISS pre-filtering")
        assignments = _cluster_with_faiss(vectors, embeddings, eps, min_samples, faiss_nlist, faiss_nprobe)
    else:
        LOGGER.info("Clustering with sklearn DBSCAN")
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = clusterer.fit_predict(vectors)
        assignments = [IdentityAssignment(embedding=emb, identity=int(label)) for emb, label in zip(embeddings, labels)]

    return assignments


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _cluster_with_faiss(
    vectors: np.ndarray,
    embeddings: Sequence[FaceEmbedding],
    eps: float,
    min_samples: int,
    nlist: int,
    nprobe: int,
) -> List[IdentityAssignment]:
    num_points = len(vectors)
    if num_points == 0:
        return []

    nlist = max(1, min(nlist, num_points))
    nprobe = max(1, min(nprobe, nlist))

    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe

    # Approximate DBSCAN: for each vector, find neighbours within eps converted to cosine similarity
    cosine_threshold = 1 - eps
    neighbours: Dict[int, List[int]] = {}
    k = min(2048, len(vectors))
    sims, idx = index.search(vectors, k)
    for i, (row_sim, row_idx) in enumerate(zip(sims, idx)):
        mask = row_sim >= cosine_threshold
        neighbourhood = [int(j) for j in row_idx[mask] if j >= 0]
        neighbours[i] = neighbourhood

    visited = set()
    labels = np.full(len(vectors), -1, dtype=int)
    cluster_id = 0

    for i in range(len(vectors)):
        if i in visited:
            continue
        visited.add(i)
        neighbour_ids = neighbours.get(i, [])
        if len(neighbour_ids) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = [id_ for id_ in neighbour_ids if id_ not in visited]
        while queue:
            j = queue.pop()
            if j not in visited:
                visited.add(j)
                labels[j] = cluster_id
                more = neighbours.get(j, [])
                if len(more) >= min_samples:
                    queue.extend([id_ for id_ in more if id_ not in visited])
        cluster_id += 1

    return [IdentityAssignment(embedding=emb, identity=int(label)) for emb, label in zip(embeddings, labels)]
