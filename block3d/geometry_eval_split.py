from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from block3d.evaluation import summarize_evaluation_records


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records_path = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected JSON object records in {records_path}, got {type(payload)}"
            )
        records.append(payload)
    return records


def metric_means_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return {
        str(key): float(value["mean"])
        for key, value in metrics.items()
        if isinstance(value, dict) and "mean" in value
    }


def summarize_geometry_eval_event_splits(
    geometry_eval_event: dict[str, Any],
) -> dict[str, Any]:
    records_path = geometry_eval_event.get("records_path")
    if records_path in (None, ""):
        raise ValueError("geometry_eval_event does not contain a records_path")

    records = sorted(
        load_jsonl_records(str(records_path)),
        key=lambda record: int(record.get("sample_idx", 0)),
    )
    total_records = len(records)

    fixed_count = int(geometry_eval_event.get("fixed_sample_count") or 0)
    random_count = int(geometry_eval_event.get("random_sample_count") or 0)
    if fixed_count < 0 or random_count < 0:
        raise ValueError(
            f"Invalid split counts: fixed_sample_count={fixed_count}, "
            f"random_sample_count={random_count}"
        )

    if fixed_count + random_count == 0:
        fixed_count = total_records
        random_count = 0
    elif fixed_count + random_count > total_records:
        raise ValueError(
            f"Split counts exceed available records: fixed={fixed_count}, "
            f"random={random_count}, records={total_records}"
        )

    fixed_records = records[:fixed_count]
    random_records = records[fixed_count : fixed_count + random_count]

    fixed_summary = summarize_evaluation_records(fixed_records) if fixed_records else None
    random_summary = (
        summarize_evaluation_records(random_records) if random_records else None
    )

    return {
        "total_records": total_records,
        "fixed_records": fixed_records,
        "random_records": random_records,
        "fixed_summary": fixed_summary,
        "random_summary": random_summary,
        "fixed_metric_means": (
            {} if fixed_summary is None else metric_means_from_summary(fixed_summary)
        ),
        "random_metric_means": (
            {} if random_summary is None else metric_means_from_summary(random_summary)
        ),
    }
