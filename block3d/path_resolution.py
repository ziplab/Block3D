from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _anchor_bases(anchor_path: str | Path | None) -> list[Path]:
    if anchor_path is None:
        return []

    anchor = Path(anchor_path).expanduser().resolve()
    base = anchor if anchor.exists() and anchor.is_dir() else anchor.parent
    bases: list[Path] = []
    for candidate in (base, *base.parents):
        bases.append(candidate)
        if candidate == REPO_ROOT:
            break
    return _unique_paths(bases)


@lru_cache(maxsize=1)
def _known_asset_roots() -> tuple[Path, ...]:
    roots: list[Path] = [
        REPO_ROOT,
        REPO_ROOT / "data",
    ]
    data_root = REPO_ROOT / "data"
    if data_root.is_dir():
        for child in data_root.iterdir():
            if child.is_dir():
                roots.append(child)
    return tuple(_unique_paths(roots))


def resolve_local_data_path(
    raw_path: str | Path,
    *,
    anchor_path: str | Path | None = None,
) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidate_paths: list[Path] = []
    anchor = None if anchor_path is None else Path(anchor_path).expanduser().resolve()
    if anchor is not None and anchor.exists() and anchor.is_file() and len(anchor.parents) >= 2:
        asset_root_base = anchor.parent.parent
        first_part = path.parts[0] if path.parts else ""
        if first_part in {anchor.parent.name, "pairs"}:
            candidate_paths.append(asset_root_base / path)

    for base in _anchor_bases(anchor_path):
        candidate_paths.append(base / path)

    for base in _known_asset_roots():
        candidate_paths.append(base / path)

    unique_candidates = _unique_paths(candidate_paths)
    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    if unique_candidates:
        return unique_candidates[0]
    return (REPO_ROOT / path).resolve()
