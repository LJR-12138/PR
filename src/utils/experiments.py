"""Experiment bookkeeping utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.config import Config
from src.utils.paths import resolve_path


def log_detector_run(cfg: Config, run_dir: Path, best_ckpt: Path) -> Path:
    """Persist detector training metadata for later comparison."""
    repo_root = resolve_path(cfg.get("paths.repo_root", "."), cfg.source.parent)
    log_dir = repo_root / "outputs" / "experiments"
    log_dir.mkdir(parents=True, exist_ok=True)

    results_csv = run_dir / "results.csv"
    metrics: Dict[str, Any] = {}
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        if not df.empty:
            final_row = df.iloc[-1].to_dict()
            metrics = {key: _to_builtin(value) for key, value in final_row.items()}
            metrics["fitness_max"] = _to_builtin(df["fitness"].max()) if "fitness" in df else None

    detector_cfg = cfg.get("training.detector", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    record = {
        "timestamp": timestamp,
        "run_dir": run_dir.as_posix(),
        "best_checkpoint": best_ckpt.as_posix(),
        "project": cfg.get("project.name", "furseal-face-intel"),
        "device": cfg.get("project.device", "cuda"),
        "detector_config": detector_cfg,
        "metrics": metrics,
    }

    record_path = log_dir / f"detector_run_{timestamp}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if results_csv.exists():
        archive_csv = log_dir / f"detector_run_{timestamp}_results.csv"
        archive_csv.write_bytes(results_csv.read_bytes())

    history_path = log_dir / "detector_history.jsonl"
    with history_path.open("a", encoding="utf-8") as history:
        history.write(json.dumps(record) + "\n")

    return record_path


def _to_builtin(value: Any) -> Any:
    """Convert numpy/pandas scalars into native Python types."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - best effort conversion
            return value
    return value
