#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


SOURCE_MESH_PREFIXES = (
    "objaverse_xl_github",
    "objaverse_xl_sketchfab",
    "abo",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare portable train/val manifests for block-diffusion training while "
            "excluding a fixed external evaluation set to avoid data leakage."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root of the portable paired dataset repo.",
    )
    parser.add_argument(
        "--source-manifest-path",
        type=str,
        required=True,
        help="Portable pair manifest used as the source training pool.",
    )
    parser.add_argument(
        "--eval-manifest-path",
        type=str,
        required=True,
        help="Prompt+bbox+mesh benchmark manifest for the fixed eval100 set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory that will receive train/val/eval manifests and the split summary.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01,
        help="Validation split ratio applied after excluding the fixed eval set.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help=(
            "Optional number of training records to select after eval exclusion and "
            "validation splitting. The selection is deterministic under --seed."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for the train/val split.",
    )
    parser.add_argument(
        "--expected-eval-count",
        type=int,
        default=100,
        help="Expected number of fixed eval records after loading the eval manifest.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Allow overwriting existing outputs in --output-dir.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object rows in {path}, got {type(payload)}")
        records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return path


def _normalize_pair_path(raw_path: Any, dataset_root: Path) -> str | None:
    if raw_path in (None, ""):
        return None

    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        try:
            return resolved.relative_to(dataset_root.resolve()).as_posix()
        except ValueError:
            parts = resolved.parts
            if "pairs" in parts:
                return Path(*parts[parts.index("pairs") :]).as_posix()
            if "records" in parts:
                return Path(*parts[parts.index("records") :]).as_posix()
            return resolved.as_posix()
    return path.as_posix()


def _normalize_source_mesh_path(raw_path: Any, dataset_root: Path) -> str | None:
    if raw_path in (None, ""):
        return None

    path = Path(str(raw_path)).expanduser()
    candidate_parts = path.parts if not path.is_absolute() else path.resolve().parts
    for prefix in SOURCE_MESH_PREFIXES:
        if prefix in candidate_parts:
            return Path(*candidate_parts[candidate_parts.index(prefix) :]).as_posix()

    if path.is_absolute():
        resolved = path.resolve()
        try:
            return resolved.relative_to(dataset_root.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()
    return path.as_posix()


def _record_pair_key(record: dict[str, Any], dataset_root: Path) -> str | None:
    raw_path = record.get("pair_path")
    if raw_path in (None, ""):
        raw_path = record.get("pair_record_key")
    return _normalize_pair_path(raw_path, dataset_root)


def _record_mesh_key(record: dict[str, Any], dataset_root: Path) -> str | None:
    if record.get("original_mesh_path") not in (None, ""):
        return _normalize_source_mesh_path(record.get("original_mesh_path"), dataset_root)
    raw_path = record.get("mesh_path")
    if raw_path in (None, ""):
        raw_path = record.get("target_mesh_path")
    return _normalize_source_mesh_path(raw_path, dataset_root)


def _record_matches_eval_source(
    source_record: dict[str, Any],
    eval_record: dict[str, Any],
    *,
    dataset_root: Path,
) -> bool:
    eval_pair_key = _record_pair_key(eval_record, dataset_root)
    if eval_pair_key is not None:
        return _record_pair_key(source_record, dataset_root) == eval_pair_key

    eval_mesh_key = _record_mesh_key(eval_record, dataset_root)
    if eval_mesh_key is not None:
        return _record_mesh_key(source_record, dataset_root) == eval_mesh_key
    return False


def _split_records(
    records: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
    train_count: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if train_count is not None and train_count <= 0:
        raise ValueError(f"train_count must be positive when provided, got {train_count}")
    if (val_ratio == 0.0 or len(records) < 2) and train_count is None:
        return list(records), []

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if val_ratio == 0.0 or len(shuffled) < 2:
        val_size = 0
    else:
        val_size = max(1, int(len(shuffled) * val_ratio))
        val_size = min(val_size, len(shuffled) - 1)

    train_candidates = shuffled[val_size:]
    if train_count is not None:
        if len(train_candidates) < train_count:
            raise ValueError(
                "Not enough records remain for the requested training subset: "
                f"requested={train_count} available={len(train_candidates)}"
            )
        train_candidates = train_candidates[:train_count]
    return train_candidates, shuffled[:val_size]


def _overlap_size(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    *,
    dataset_root: Path,
) -> int:
    def _identity_key(record: dict[str, Any]) -> str | None:
        pair_key = _record_pair_key(record, dataset_root)
        if pair_key is not None:
            return f"pair:{pair_key}"
        item_id = record.get("item_id")
        if item_id not in (None, ""):
            return f"item:{item_id}"
        mesh_key = _record_mesh_key(record, dataset_root)
        if mesh_key is not None:
            return f"mesh:{mesh_key}"
        return None

    keys_a = {key for record in records_a if (key := _identity_key(record)) is not None}
    keys_b = {key for record in records_b if (key := _identity_key(record)) is not None}
    return len(keys_a & keys_b)


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    source_manifest_path = Path(args.source_manifest_path).expanduser().resolve()
    eval_manifest_path = Path(args.eval_manifest_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise ValueError(
            f"Output directory {output_dir} already exists and is not empty. "
            "Pass --overwrite to reuse it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_records = _read_jsonl(source_manifest_path)
    eval_records = _read_jsonl(eval_manifest_path)

    if args.expected_eval_count > 0 and len(eval_records) != int(args.expected_eval_count):
        raise ValueError(
            f"Expected {args.expected_eval_count} eval records, found {len(eval_records)} "
            f"in {eval_manifest_path}"
        )

    source_pair_keys = {
        key
        for record in source_records
        if (key := _record_pair_key(record, dataset_root)) is not None
    }
    source_mesh_keys = {
        key
        for record in source_records
        if (key := _record_mesh_key(record, dataset_root)) is not None
    }

    eval_pair_keys = {
        key for record in eval_records if (key := _record_pair_key(record, dataset_root)) is not None
    }
    eval_mesh_keys = {
        key for record in eval_records if (key := _record_mesh_key(record, dataset_root)) is not None
    }

    missing_eval_records: list[dict[str, Any]] = []
    for record in eval_records:
        pair_key = _record_pair_key(record, dataset_root)
        mesh_key = _record_mesh_key(record, dataset_root)
        matched = False
        if pair_key is not None:
            matched = pair_key in source_pair_keys
        elif mesh_key is not None:
            matched = mesh_key in source_mesh_keys
        if not matched:
            missing_eval_records.append(record)
    if missing_eval_records:
        missing_ids = [str(record.get("sample_id", idx)) for idx, record in enumerate(missing_eval_records)]
        raise ValueError(
            "Some eval records do not match the source manifest and cannot be safely excluded: "
            f"{missing_ids}"
        )

    excluded_source_records: list[dict[str, Any]] = []
    remaining_source_records: list[dict[str, Any]] = []
    for record in source_records:
        should_exclude = any(
            _record_matches_eval_source(record, eval_record, dataset_root=dataset_root)
            for eval_record in eval_records
        )
        if should_exclude:
            excluded_source_records.append(record)
        else:
            remaining_source_records.append(record)

    if len(excluded_source_records) != len(eval_records):
        raise ValueError(
            "Excluded source record count does not match eval record count: "
            f"excluded={len(excluded_source_records)} eval={len(eval_records)}"
        )

    train_records, val_records = _split_records(
        remaining_source_records,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        train_count=args.train_count,
    )

    train_manifest_path = _write_jsonl(output_dir / "train_manifest.jsonl", train_records)
    val_manifest_path = _write_jsonl(output_dir / "val_manifest.jsonl", val_records)
    eval_output_path = _write_jsonl(output_dir / "eval100_manifest.jsonl", eval_records)
    excluded_output_path = _write_jsonl(
        output_dir / "excluded_eval100_source_records.jsonl",
        excluded_source_records,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "source_manifest_path": str(source_manifest_path),
        "eval_manifest_path": str(eval_manifest_path),
        "train_manifest_path": str(train_manifest_path),
        "val_manifest_path": str(val_manifest_path),
        "eval100_manifest_path": str(eval_output_path),
        "excluded_eval100_source_records_path": str(excluded_output_path),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "requested_train_count": args.train_count,
        "source_record_count": len(source_records),
        "eval_record_count": len(eval_records),
        "excluded_source_record_count": len(excluded_source_records),
        "remaining_source_record_count": len(remaining_source_records),
        "train_candidate_record_count": len(remaining_source_records) - len(val_records),
        "selected_train_record_count": len(train_records),
        "train_record_count": len(train_records),
        "val_record_count": len(val_records),
        "source_eval_pair_overlap_count": len(source_pair_keys & eval_pair_keys),
        "source_eval_mesh_overlap_count": len(source_mesh_keys & eval_mesh_keys),
        "train_val_overlap_count": _overlap_size(
            train_records,
            val_records,
            dataset_root=dataset_root,
        ),
        "train_eval_overlap_count": _overlap_size(
            train_records,
            eval_records,
            dataset_root=dataset_root,
        ),
        "val_eval_overlap_count": _overlap_size(
            val_records,
            eval_records,
            dataset_root=dataset_root,
        ),
    }
    _write_json(output_dir / "split_summary.json", summary)

    if summary["train_val_overlap_count"] != 0:
        raise RuntimeError(f"train/val overlap detected: {summary['train_val_overlap_count']}")
    if summary["train_eval_overlap_count"] != 0:
        raise RuntimeError(
            f"train/eval overlap detected: {summary['train_eval_overlap_count']}"
        )
    if summary["val_eval_overlap_count"] != 0:
        raise RuntimeError(f"val/eval overlap detected: {summary['val_eval_overlap_count']}")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
