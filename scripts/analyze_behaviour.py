"""Generate behaviour analytics reports from identity assignments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd
import tyro
import torch

from src.analysis.behaviour import build_sightings, cooccurrence_graph, summarise_population
from src.recognition.cluster import IdentityAssignment
from src.recognition.embedding_model import FaceEmbedding
from src.recognition.face_extractor import FacePatch
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
    embeddings_by_patch = {emb.face.patch_path.as_posix(): emb for emb in embeddings}

    identities_path = embeddings_path.with_name("identities.parquet")
    if not identities_path.exists():
        raise FileNotFoundError("Run cluster_identities.py first to create identities.parquet")
    df_identities = pd.read_parquet(identities_path)

    assignments = _load_assignments(df_identities, embeddings_by_patch)
    sightings = build_sightings(assignments)

    report_dir = resolve_path(cfg.get("paths.behaviour_report_dir", "outputs/reports"), repo_root)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = summarise_population(sightings)
    summary_path = report_dir / "population_summary.csv"
    summary.to_csv(summary_path, index=False)

    graph = cooccurrence_graph(sightings, min_cooccurrence=cfg.get("behaviour.min_cooccurrence", 2))
    if graph.number_of_nodes() > 0:
        graph_path = report_dir / "cooccurrence.graphml"
        nx.write_graphml(graph, graph_path)
    else:
        LOGGER.warning("Co-occurrence graph empty; skipping GraphML export")

    html_path = report_dir / "summary.html"
    _write_html_report(summary, graph, html_path)
    LOGGER.info("Behaviour reports saved under %s", report_dir)


def _load_assignments(df: pd.DataFrame, embeddings_by_patch: dict[str, FaceEmbedding]) -> list[IdentityAssignment]:
    assignments: list[IdentityAssignment] = []
    for row in df.itertuples(index=False):
        patch_path = row.patch_path
        embedding = embeddings_by_patch.get(patch_path)
        if embedding is None:
            face = FacePatch(
                image_path=Path(row.image_path),
                patch_path=Path(row.patch_path),
                bbox_xyxy=(float(row.bbox_x1), float(row.bbox_y1), float(row.bbox_x2), float(row.bbox_y2)),
                score=float(row.score),
            )
            embedding = FaceEmbedding(face=face, embedding=torch.zeros(512, dtype=torch.float32))
        assignments.append(IdentityAssignment(embedding=embedding, identity=int(row.identity)))
    return assignments


def _write_html_report(summary: pd.DataFrame, graph: nx.Graph, path: Path) -> None:
    html = ["<html><head><title>Fur Seal Behaviour Summary</title></head><body>"]
    html.append("<h1>Population Summary</h1>")
    if summary.empty:
        html.append("<p>No identities clustered yet.</p>")
    else:
        html.append(summary.to_html(index=False))

    html.append("<h2>Co-occurrence Graph</h2>")
    if graph.number_of_edges() == 0:
        html.append("<p>No co-occurrences above threshold.</p>")
    else:
        html.append("<ul>")
        for u, v, data in graph.edges(data=True):
            html.append(f"<li>Identity {u} ↔ Identity {v}: weight={data.get('weight', 0)}</li>")
        html.append("</ul>")
    html.append("</body></html>")
    path.write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    main(tyro.cli(Args))
