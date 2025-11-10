"""Behaviour informatics for fur-seal sightings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import networkx as nx
import pandas as pd

from src.recognition.cluster import IdentityAssignment

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SightRecord:
    identity: int
    image_path: Path
    timestamp: pd.Timestamp
    latitude: float | None
    longitude: float | None


def build_sightings(assignments: Iterable[IdentityAssignment]) -> List[SightRecord]:
    """Merge identity assignments with EXIF metadata."""
    records: List[SightRecord] = []
    for assignment in assignments:
        face = assignment.embedding.face
        meta = _read_exif(face.image_path)
        records.append(
            SightRecord(
                identity=assignment.identity,
                image_path=face.image_path,
                timestamp=meta.get("timestamp", pd.NaT),
                latitude=meta.get("latitude"),
                longitude=meta.get("longitude"),
            )
        )
    return records


def cooccurrence_graph(records: Iterable[SightRecord], min_cooccurrence: int = 2) -> nx.Graph:
    """Build a co-occurrence graph by image-level co-presence."""
    df = pd.DataFrame(
        [
            {
                "identity": rec.identity,
                "image": rec.image_path.as_posix(),
            }
            for rec in records
            if rec.identity >= 0
        ]
    )
    if df.empty:
        return nx.Graph()
    grouped = df.groupby("image")
    graph = nx.Graph()
    for _, group in grouped:
        ids = group["identity"].unique()
        for identity in ids:
            graph.add_node(int(identity))
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1 :]:
                if graph.has_edge(int(id_a), int(id_b)):
                    graph[int(id_a)][int(id_b)]["weight"] += 1
                else:
                    graph.add_edge(int(id_a), int(id_b), weight=1)
    to_remove = [edge for edge, attrs in graph.edges.items() if attrs.get("weight", 0) < min_cooccurrence]
    graph.remove_edges_from(to_remove)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def summarise_population(records: Iterable[SightRecord]) -> pd.DataFrame:
    """Produce per-identity statistics."""
    df = pd.DataFrame(
        [
            {
                "identity": rec.identity,
                "timestamp": rec.timestamp,
                "latitude": rec.latitude,
                "longitude": rec.longitude,
            }
            for rec in records
            if rec.identity >= 0
        ]
    )
    if df.empty:
        return pd.DataFrame(columns=["identity", "sightings", "first_seen", "last_seen"])

    summary = (
        df.groupby("identity")
        .agg(
            sightings=("identity", "count"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        .reset_index()
    )
    return summary


def _read_exif(image_path: Path) -> dict[str, Optional[float] | pd.Timestamp]:
    """Extract timestamp and GPS coordinates from EXIF data."""
    import exifread

    tags = {}
    with image_path.open("rb") as fobj:
        tags = exifread.process_file(fobj, details=False)

    timestamp = None
    for tag in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
        if tag in tags:
            timestamp = pd.to_datetime(str(tags[tag]), errors="coerce")
            break

    def _parse_coord(prefix: str) -> Optional[float]:
        ref_tag = f"GPS {prefix}Ref"
        coord_tag = f"GPS {prefix}"
        if ref_tag not in tags or coord_tag not in tags:
            return None
        ref = str(tags[ref_tag])
        values = tags[coord_tag].values
        degrees = float(values[0].num) / float(values[0].den)
        minutes = float(values[1].num) / float(values[1].den)
        seconds = float(values[2].num) / float(values[2].den)
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ("S", "W"):
            decimal *= -1
        return decimal

    return {
        "timestamp": timestamp,
        "latitude": _parse_coord("GPSLatitude"),
        "longitude": _parse_coord("GPSLongitude"),
    }
