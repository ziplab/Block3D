from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from block3d.path_resolution import resolve_local_data_path


@dataclass(frozen=True)
class BenchmarkPrompt:
    sample_id: str
    prompt_text: str
    bbox_xyz: list[float]
    mesh_path: str
    pair_path: str | None = None


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    if lower_idx == upper_idx:
        return sorted_values[lower_idx]
    frac = position - lower_idx
    lower = sorted_values[lower_idx]
    upper = sorted_values[upper_idx]
    return lower * (1.0 - frac) + upper * frac


def _load_benchmark_prompt_records(path: Path) -> list[dict[str, Any]]:
    raw_text = path.read_text()
    stripped = raw_text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(raw_text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list payload in {path}, got {type(payload)}")
        return [record for record in payload if isinstance(record, dict)]

    records: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object records in {path}, got {type(payload)}")
        records.append(payload)
    return records


def load_benchmark_prompts(path: str | Path) -> list[BenchmarkPrompt]:
    resolved_path = Path(path).expanduser().resolve()
    prompts: list[BenchmarkPrompt] = []
    for record in _load_benchmark_prompt_records(resolved_path):
        mesh_path = (
            record.get("mesh_path")
            or record.get("target_mesh_path")
            or record.get("reference_mesh_path")
        )
        if mesh_path in (None, ""):
            raise ValueError(
                "Benchmark record is missing mesh_path, target_mesh_path, or "
                f"reference_mesh_path: {record.get('sample_id', '<unknown>')}"
            )
        pair_path = record.get("pair_path") or record.get("pair_record_key")
        prompts.append(
            BenchmarkPrompt(
                sample_id=str(record["sample_id"]),
                prompt_text=str(record["prompt_text"]),
                bbox_xyz=[float(value) for value in record["bbox_xyz"]],
                mesh_path=str(
                    resolve_local_data_path(mesh_path, anchor_path=resolved_path)
                ),
                pair_path=(
                    None
                    if pair_path in (None, "")
                    else str(
                        resolve_local_data_path(pair_path, anchor_path=resolved_path)
                    )
                ),
            )
        )
    return prompts


def summarize_numeric_series(values: Iterable[float]) -> dict[str, float]:
    series = [float(value) for value in values]
    if not series:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(statistics.fmean(series)),
        "median": float(statistics.median(series)),
        "p90": _quantile(series, 0.90),
        "std": float(statistics.pstdev(series)) if len(series) > 1 else 0.0,
        "min": float(min(series)),
        "max": float(max(series)),
    }
