from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import trimesh

NATIVE_MESH_SUFFIXES = {".obj", ".ply", ".glb", ".gltf", ".stl", ".off"}
BLENDER_MESH_SUFFIXES = {".fbx", ".dae", ".blend"}
BLENDER_IMPORTABLE_SUFFIXES = {
    ".obj",
    ".ply",
    ".glb",
    ".gltf",
    ".stl",
    ".fbx",
    ".dae",
    ".blend",
}
MESH_SUFFIXES = NATIVE_MESH_SUFFIXES | BLENDER_MESH_SUFFIXES
TEXT_KEYS = ("text", "caption", "prompt", "description", "name", "title")
MESH_KEYS = ("mesh_path", "path", "file_path", "mesh", "object_path")
PAIR_KEYS = ("pair_path", "paired_path", "token_pair_path")
BOUNDING_BOX_MAX_SIZE = 1.925
MESH_SCALE = 0.96
CACHE_VERSION = 2
MESH_IO_DONTNEED_ENABLED = os.environ.get("BLOCK3D_MESH_IO_DONTNEED", "1") != "0"
BLENDER_CACHE_ROOT = (
    Path(os.environ.get("BLOCK3D_BLENDER_MESH_CACHE", "~/.cache/block3d/blender_mesh_cache"))
    .expanduser()
    .resolve()
)


@dataclass
class DatasetDiscoverySummary:
    total_records: int = 0
    valid_entries: int = 0
    selected_entries: int = 0
    missing_mesh: int = 0
    missing_pair: int = 0
    unsupported_entries: int = 0
    manifest_text_entries: int = 0
    sidecar_text_entries: int = 0
    stem_text_entries: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class SampleEvalSpec:
    prompt_text: str
    bbox_xyz: list[float]
    mesh_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_bbox(bounding_box_xyz: tuple[float, float, float]) -> list[float]:
    max_length = max(bounding_box_xyz)
    if max_length <= 0:
        return [0.0, 0.0, 0.0]
    return [BOUNDING_BOX_MAX_SIZE * elem / max_length for elem in bounding_box_xyz]


def rescale(vertices: np.ndarray, mesh_scale: float = MESH_SCALE) -> np.ndarray:
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    return (vertices - center) * scale


def _resolve_blender_binary() -> str:
    blender_bin = os.environ.get("BLOCK3D_BLENDER_BIN", "blender")
    resolved = shutil.which(blender_bin)
    if resolved is None:
        raise RuntimeError(
            "Blender is required for mesh conversion fallback but was not found in PATH. "
            "Install blender or set BLOCK3D_BLENDER_BIN."
        )
    return resolved


def _blender_script_path() -> Path:
    return Path(__file__).resolve().with_name("blender_convert_mesh.py")


def _converted_mesh_cache_path(mesh_path: Path) -> Path:
    stat = mesh_path.stat()
    cache_key = hashlib.sha256(
        f"{mesh_path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()
    return BLENDER_CACHE_ROOT / cache_key[:2] / cache_key / "converted.obj"


def _convert_mesh_with_blender(mesh_path: Path) -> Path:
    cached_output = _converted_mesh_cache_path(mesh_path)
    if cached_output.exists():
        return cached_output

    cached_output.parent.mkdir(parents=True, exist_ok=True)
    blender_bin = _resolve_blender_binary()
    script_path = _blender_script_path()

    with tempfile.TemporaryDirectory(
        prefix="block3d_blender_", dir=str(cached_output.parent)
    ) as tmp_dir:
        tmp_output = Path(tmp_dir) / "converted.obj"
        command = [
            blender_bin,
            "--background",
            "-noaudio",
            "--python",
            str(script_path),
            "--",
            "--input",
            str(mesh_path),
            "--output",
            str(tmp_output),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            raise RuntimeError(
                "Blender mesh conversion failed for "
                f"{mesh_path}: returncode={completed.returncode}, "
                f"stdout={stdout!r}, stderr={stderr!r}"
            )
        if not tmp_output.exists():
            raise RuntimeError(
                f"Blender mesh conversion did not produce output for {mesh_path}"
            )
        shutil.move(str(tmp_output), str(cached_output))

    return cached_output


def _load_trimesh(mesh_path: Path) -> "trimesh.Trimesh":
    import trimesh

    # Geometry tokenization only depends on vertices/faces, so skip trimesh's
    # eager processing and material/texture resolution to reduce I/O overhead.
    return trimesh.load(
        str(mesh_path),
        force="mesh",
        process=False,
        skip_materials=True,
    )


def _drop_file_cache_hint(path: Path) -> None:
    if not MESH_IO_DONTNEED_ENABLED or not hasattr(os, "posix_fadvise"):
        return
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advice is None:
        return
    try:
        if not path.exists() or not path.is_file():
            return
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, advice)
        finally:
            os.close(fd)
    except Exception:
        # This is only an advisory optimization. Ignore unsupported filesystems
        # and transient races to avoid breaking the preprocessing pipeline.
        return


def _iter_gltf_buffer_uris(mesh_path: Path) -> list[str]:
    try:
        payload = json.loads(mesh_path.read_text())
    except Exception:
        return []

    buffers = payload.get("buffers")
    if not isinstance(buffers, list):
        return []

    resolved: list[str] = []
    for entry in buffers:
        if not isinstance(entry, dict):
            continue
        uri = entry.get("uri")
        if not isinstance(uri, str):
            continue
        stripped = uri.strip()
        if not stripped or stripped.startswith("data:") or "://" in stripped:
            continue
        resolved.append(stripped)
    return resolved


def _mesh_dependency_paths(mesh_path: Path) -> list[Path]:
    paths = [mesh_path]
    if mesh_path.suffix.lower() != ".gltf":
        return paths
    for uri in _iter_gltf_buffer_uris(mesh_path):
        dependency_path = mesh_path.parent / uri
        if dependency_path.exists():
            paths.append(dependency_path)
    return paths


def _ensure_gltf_buffer_dependencies(mesh_path: Path) -> None:
    if mesh_path.suffix.lower() != ".gltf":
        return
    missing: list[Path] = []
    for uri in _iter_gltf_buffer_uris(mesh_path):
        target_path = mesh_path.parent / uri
        if not target_path.exists():
            missing.append(target_path)
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing local glTF buffer dependencies for {mesh_path}: {missing_text}"
        )


def load_scaled_mesh(file_path: str, *, allow_blender_fallback: bool = True) -> "trimesh.Trimesh":
    import trimesh

    mesh_path = Path(file_path).expanduser().resolve()
    _ensure_gltf_buffer_dependencies(mesh_path)
    read_paths = _mesh_dependency_paths(mesh_path)
    converted_path: Optional[Path] = None
    try:
        mesh: trimesh.Trimesh = _load_trimesh(mesh_path)
    except Exception as original_error:
        if not allow_blender_fallback:
            raise RuntimeError(
                f"Direct mesh load failed with Blender fallback disabled for {mesh_path}: "
                f"{original_error!r}"
            ) from original_error
        suffix = mesh_path.suffix.lower()
        if suffix not in BLENDER_IMPORTABLE_SUFFIXES:
            raise
        converted_path = _convert_mesh_with_blender(mesh_path)
        try:
            mesh = _load_trimesh(converted_path)
        except Exception as converted_error:
            raise RuntimeError(
                f"Failed to load mesh {mesh_path} directly and after Blender conversion. "
                f"direct_error={original_error!r}; converted_error={converted_error!r}"
            ) from converted_error
    mesh.remove_infinite_values()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh has no vertices or faces after cleaning")
    mesh.vertices = rescale(mesh.vertices)
    for path in read_paths:
        _drop_file_cache_hint(path)
    if converted_path is not None:
        _drop_file_cache_hint(converted_path)
    return mesh


def _first_value(record: dict, keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return None


def _load_json_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            return list(data["samples"])
        return [data]
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_manifest(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_records(path)
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported manifest format: {path}")


def _read_sidecar_text(mesh_path: Path) -> tuple[Optional[str], Optional[str]]:
    txt_path = mesh_path.with_suffix(".txt")
    if txt_path.exists():
        text = txt_path.read_text().strip()
        if text:
            return text, "sidecar_txt"

    json_path = mesh_path.with_suffix(".json")
    if json_path.exists():
        try:
            record = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            record = None
        if isinstance(record, dict):
            text = _first_value(record, TEXT_KEYS)
            if text:
                return text, "sidecar_json"
    return None, None


def _summary_from_text_source(summary: DatasetDiscoverySummary, source: str) -> None:
    if source == "manifest":
        summary.manifest_text_entries += 1
    elif source.startswith("sidecar"):
        summary.sidecar_text_entries += 1
    else:
        summary.stem_text_entries += 1


def discover_objaverse_entries(
    root: str | Path,
    manifest_path: Optional[str | Path] = None,
    max_samples: Optional[int] = None,
    verify_paths: bool = True,
) -> tuple[list[dict], DatasetDiscoverySummary]:
    root_path = Path(root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve() if manifest_path else None
    summary = DatasetDiscoverySummary()
    entries: list[dict] = []

    if manifest is not None:
        records = _load_manifest(manifest)
        summary.total_records = len(records)
        for record in records:
            mesh_value = _first_value(record, MESH_KEYS)
            if mesh_value is None:
                summary.unsupported_entries += 1
                continue
            mesh_path = Path(mesh_value)
            if not mesh_path.is_absolute():
                mesh_path = (root_path / mesh_path).resolve()

            pair_path = None
            pair_value = _first_value(record, PAIR_KEYS)
            if pair_value is not None:
                resolved_pair = Path(pair_value)
                if not resolved_pair.is_absolute():
                    resolved_pair = (root_path / resolved_pair).resolve()
                if not verify_paths or resolved_pair.exists():
                    pair_path = resolved_pair
                else:
                    summary.missing_pair += 1

            mesh_exists = mesh_path.exists() if verify_paths else False
            if verify_paths and not mesh_exists:
                summary.missing_mesh += 1
                if pair_path is None:
                    continue

            text = _first_value(record, TEXT_KEYS)
            if text is None:
                if verify_paths and mesh_exists:
                    text, source = _read_sidecar_text(mesh_path)
                else:
                    text, source = None, None
            else:
                source = "manifest"
            if text is None:
                text = mesh_path.stem.replace("_", " ")
                source = "stem"

            entries.append({"mesh_path": mesh_path, "text": text, "pair_path": pair_path})
            _summary_from_text_source(summary, source)
    else:
        mesh_candidates = sorted(root_path.rglob("*"))
        for mesh_path in mesh_candidates:
            if not mesh_path.is_file():
                continue
            if mesh_path.suffix.lower() not in MESH_SUFFIXES:
                continue

            summary.total_records += 1
            text, source = _read_sidecar_text(mesh_path)
            if text is None:
                text = mesh_path.stem.replace("_", " ")
                source = "stem"

            entries.append({"mesh_path": mesh_path.resolve(), "text": text, "pair_path": None})
            _summary_from_text_source(summary, source)

    summary.valid_entries = len(entries)
    if max_samples is not None:
        entries = entries[: max(max_samples, 0)]
    summary.selected_entries = len(entries)
    return entries, summary


def split_objaverse_entries(
    entries: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    if val_ratio == 0.0 or len(entries) < 2:
        return list(entries), []

    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    val_size = min(val_size, len(shuffled) - 1)
    return shuffled[val_size:], shuffled[:val_size]


def prepare_sample_eval_specs(
    entries: list[dict],
    max_samples: int,
    bad_samples_path: Optional[str] = None,
    selection_seed: Optional[int] = None,
) -> list[SampleEvalSpec]:
    if max_samples <= 0:
        return []

    bad_samples = (
        Path(bad_samples_path).expanduser().resolve()
        if bad_samples_path is not None
        else None
    )
    if bad_samples is not None:
        bad_samples.parent.mkdir(parents=True, exist_ok=True)

    candidate_entries = list(entries)
    rng = random.Random() if selection_seed is None else random.Random(int(selection_seed))
    rng.shuffle(candidate_entries)

    specs: list[SampleEvalSpec] = []
    for entry in candidate_entries:
        if len(specs) >= max_samples:
            break

        mesh_path = Path(entry["mesh_path"]).expanduser().resolve()
        try:
            bbox = None
            pair_value = entry.get("pair_path")
            if pair_value is not None:
                pair_path = Path(pair_value).expanduser().resolve()
                if pair_path.exists():
                    bbox = _load_bbox_from_pair_file(pair_path)
            if bbox is None:
                mesh = load_scaled_mesh(str(mesh_path))
                bbox = _compute_bbox(mesh).tolist()
            specs.append(
                SampleEvalSpec(
                    prompt_text=str(entry["text"]),
                    bbox_xyz=[float(value) for value in bbox],
                    mesh_path=str(mesh_path),
                )
            )
        except Exception as error:
            if bad_samples is not None:
                with bad_samples.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "mesh_path": str(mesh_path),
                                "prompt_text": str(entry.get("text", "")),
                                "error": repr(error),
                            }
                        )
                        + "\n"
                    )
            continue
    return specs


def _sample_point_cloud(mesh: "trimesh.Trimesh", n_samples: int) -> torch.Tensor:
    import trimesh

    positions, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_indices]
    point_cloud = np.concatenate([positions, normals], axis=1).astype(np.float32)
    return torch.from_numpy(point_cloud)


def _compute_bbox(mesh: "trimesh.Trimesh") -> torch.Tensor:
    extent = mesh.vertices.max(0) - mesh.vertices.min(0)
    bbox = normalize_bbox(tuple(float(v) for v in extent))
    return torch.tensor(bbox, dtype=torch.float32)


def _load_bbox_from_pair_file(pair_path: Path) -> Optional[list[float]]:
    try:
        paired = torch.load(pair_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(paired, dict) or "bbox_xyz" not in paired:
        return None

    bbox_xyz = paired["bbox_xyz"]
    if isinstance(bbox_xyz, torch.Tensor):
        bbox_values = bbox_xyz.view(-1).tolist()
    elif isinstance(bbox_xyz, (list, tuple)):
        bbox_values = list(bbox_xyz)
    else:
        return None
    if len(bbox_values) < 3:
        return None
    return [float(bbox_values[0]), float(bbox_values[1]), float(bbox_values[2])]


class ObjaverseDataset(Dataset):
    def __init__(
        self,
        root: str,
        manifest_path: Optional[str] = None,
        point_samples: int = 8192,
        cache_dir: Optional[str] = None,
        entries: Optional[list[dict]] = None,
        bad_samples_path: Optional[str] = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve() if manifest_path else None
        )
        self.point_samples = point_samples
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        )
        self.bad_samples_path = (
            Path(bad_samples_path).expanduser().resolve()
            if bad_samples_path is not None
            else None
        )
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.bad_samples_path is not None:
            self.bad_samples_path.parent.mkdir(parents=True, exist_ok=True)

        if entries is None:
            self.entries, self.discovery_summary = discover_objaverse_entries(
                root=self.root,
                manifest_path=self.manifest_path,
            )
        else:
            self.entries = list(entries)
            self.discovery_summary = DatasetDiscoverySummary(
                total_records=len(self.entries),
                valid_entries=len(self.entries),
                selected_entries=len(self.entries),
            )

        if not self.entries:
            raise ValueError(
                f"No valid Objaverse entries found under {self.root} "
                f"(manifest={self.manifest_path})"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def _cache_path(self, mesh_path: Path) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        path_digest = hashlib.sha1(str(mesh_path).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{mesh_path.stem}_{path_digest}.pt"

    def _load_cached_sample(self, cache_path: Path, mesh_path: Path) -> Optional[dict]:
        cached = torch.load(cache_path, map_location="cpu")
        metadata = cached.get("_metadata")
        mesh_mtime_ns = mesh_path.stat().st_mtime_ns
        if metadata is None:
            return None
        if metadata.get("cache_version") != CACHE_VERSION:
            return None
        if metadata.get("point_samples") != self.point_samples:
            return None
        if metadata.get("mesh_mtime_ns") != mesh_mtime_ns:
            return None
        return {
            "point_cloud": cached["point_cloud"],
            "bbox_xyz": cached["bbox_xyz"],
        }

    def _write_bad_sample(self, entry: dict, error: Exception) -> None:
        if self.bad_samples_path is None:
            return
        payload = {
            "mesh_path": str(entry["mesh_path"]),
            "pair_path": None if entry.get("pair_path") is None else str(entry["pair_path"]),
            "prompt_text": entry["text"],
            "error": repr(error),
        }
        with self.bad_samples_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _load_paired_sample(self, pair_path: Path, entry: dict) -> dict:
        paired = torch.load(pair_path, map_location="cpu")
        required = {"bbox_xyz", "text_input_ids", "text_attention_mask", "shape_ids"}
        missing = sorted(required - set(paired))
        if missing:
            raise KeyError(f"Paired sample {pair_path} is missing keys: {missing}")

        bbox_xyz = paired["bbox_xyz"]
        text_input_ids = paired["text_input_ids"]
        text_attention_mask = paired["text_attention_mask"]
        shape_ids = paired["shape_ids"]

        if not isinstance(bbox_xyz, torch.Tensor):
            bbox_xyz = torch.tensor(bbox_xyz, dtype=torch.float32)
        if not isinstance(text_input_ids, torch.Tensor):
            text_input_ids = torch.tensor(text_input_ids, dtype=torch.long)
        if not isinstance(text_attention_mask, torch.Tensor):
            text_attention_mask = torch.tensor(text_attention_mask, dtype=torch.long)
        if not isinstance(shape_ids, torch.Tensor):
            shape_ids = torch.tensor(shape_ids, dtype=torch.long)

        return {
            "prompt_text": entry["text"],
            "bbox_xyz": bbox_xyz.view(-1).to(dtype=torch.float32),
            "text_input_ids": text_input_ids.view(-1).to(dtype=torch.long),
            "text_attention_mask": text_attention_mask.view(-1).to(dtype=torch.long),
            "shape_ids": shape_ids.view(-1).to(dtype=torch.long),
            "mesh_path": str(entry["mesh_path"]),
            "pair_path": str(pair_path),
        }

    def __getitem__(self, idx: int) -> Optional[dict]:
        entry = self.entries[idx]
        mesh_path = entry["mesh_path"]
        pair_path = entry.get("pair_path")
        cache_path = self._cache_path(mesh_path)

        try:
            if pair_path is not None:
                return self._load_paired_sample(pair_path, entry)

            if cache_path is not None and cache_path.exists():
                cached = self._load_cached_sample(cache_path, mesh_path)
                if cached is not None:
                    cached["prompt_text"] = entry["text"]
                    cached["mesh_path"] = str(mesh_path)
                    return cached

            mesh = load_scaled_mesh(str(mesh_path))
            sample = {
                "prompt_text": entry["text"],
                "point_cloud": _sample_point_cloud(mesh, self.point_samples),
                "bbox_xyz": _compute_bbox(mesh),
                "mesh_path": str(mesh_path),
            }
            if cache_path is not None:
                torch.save(
                    {
                        "_metadata": {
                            "cache_version": CACHE_VERSION,
                            "mesh_mtime_ns": mesh_path.stat().st_mtime_ns,
                            "point_samples": self.point_samples,
                        },
                        "point_cloud": sample["point_cloud"],
                        "bbox_xyz": sample["bbox_xyz"],
                    },
                    cache_path,
                )
            return sample
        except Exception as error:
            self._write_bad_sample(entry, error)
            return None


def collate_objaverse_batch(samples: list[Optional[dict]]) -> Optional[dict]:
    valid_samples = [sample for sample in samples if sample is not None]
    if not valid_samples:
        return None

    batch = {
        "prompt_text": [sample["prompt_text"] for sample in valid_samples],
        "bbox_xyz": torch.stack([sample["bbox_xyz"] for sample in valid_samples], dim=0),
        "mesh_path": [sample["mesh_path"] for sample in valid_samples],
    }

    if "point_cloud" in valid_samples[0]:
        batch["point_cloud"] = torch.stack(
            [sample["point_cloud"] for sample in valid_samples], dim=0
        )
    if "shape_ids" in valid_samples[0]:
        batch["shape_ids"] = torch.stack(
            [sample["shape_ids"] for sample in valid_samples], dim=0
        )
        batch["text_input_ids"] = torch.stack(
            [sample["text_input_ids"] for sample in valid_samples], dim=0
        )
        batch["text_attention_mask"] = torch.stack(
            [sample["text_attention_mask"] for sample in valid_samples], dim=0
        )
        batch["pair_path"] = [sample["pair_path"] for sample in valid_samples]
    return batch
