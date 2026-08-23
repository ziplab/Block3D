from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from scipy.spatial import cKDTree

from block3d.benchmarking import summarize_numeric_series
from block3d.path_resolution import resolve_local_data_path
from block3d.training.data import load_scaled_mesh, normalize_bbox

DEFAULT_FSCORE_THRESHOLD_PCTS = (0.01, 0.02)
EVALUATION_VERSION = 2
NUMERIC_METRIC_KEYS = (
    "cd_l1",
    "normal_consistency",
)
CLIP_NUMERIC_METRIC_KEYS = (
    "clipscore_mean",
    "clipscore_max",
    "clipscore_min",
    "clipscore_std",
    "clipscore_cosine_mean",
    "clipscore_cosine_max",
    "clipscore_cosine_min",
    "clipscore_cosine_std",
    "mv_clip_score_mean",
    "mv_clip_score_max",
    "mv_clip_score_min",
    "mv_clip_score_std",
)
CLIP_RECORD_KEYS = set(CLIP_NUMERIC_METRIC_KEYS) | {
    "clipscore_model_name",
    "clipscore_num_views",
    "clipscore_view_scores",
    "clipscore_cosine_view_scores",
    "mv_clip_model_name",
    "mv_clip_num_views",
    "mv_clip_view_scores",
    "mv_clip_status",
    "mv_clip_error",
    "mv_clip_render_backend",
    "mv_clip_render_dir",
    "mv_clip_image_paths",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object records in {path}, got {type(payload)}")
        records.append(payload)
    return records


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True, default=str) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def save_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen = set()
    for record in records:
        for key in record.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_run_spec(spec: str) -> tuple[Optional[str], Path]:
    if "=" not in spec:
        return None, Path(spec).expanduser().resolve()
    method, raw_path = spec.split("=", 1)
    method = method.strip()
    return (method or None), Path(raw_path).expanduser().resolve()


def _derive_method_name(path: Path, explicit_method: Optional[str]) -> str:
    if explicit_method:
        return explicit_method
    if path.is_file():
        return path.stem
    return path.name


def _load_records_from_dir(path: Path) -> list[dict[str, Any]]:
    samples_jsonl = path / "samples.jsonl"
    if samples_jsonl.exists():
        return _read_jsonl(samples_jsonl)

    summary_json = path / "summary.json"
    if summary_json.exists():
        payload = _read_json(summary_json)
        samples = payload.get("samples")
        if isinstance(samples, list):
            return [dict(sample) for sample in samples if isinstance(sample, dict)]

    metadata_paths = sorted(path.rglob("metadata.json"))
    if metadata_paths:
        return [_read_json(metadata_path) for metadata_path in metadata_paths]

    raise ValueError(
        f"Could not discover evaluation records under {path}. "
        "Expected samples.jsonl, summary.json, or per-sample metadata.json files."
    )


def _coerce_bbox(value: Any) -> Optional[list[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return None


def _default_generation_status(record: dict[str, Any]) -> str:
    status = record.get("status")
    if status is not None:
        return str(status)
    if record.get("generated_mesh_path"):
        return "ok"
    return "missing"


def _sample_key(record: dict[str, Any], idx: int) -> str:
    for key in ("sample_key", "sample_id", "item_id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    sample_dir = record.get("sample_dir")
    if sample_dir:
        return Path(str(sample_dir)).name
    sample_idx = record.get("sample_idx")
    if sample_idx is not None:
        return f"{int(sample_idx):04d}"
    return f"{idx:04d}"


def load_evaluation_records(
    path: str | Path,
    method: Optional[str] = None,
) -> list[dict[str, Any]]:
    resolved_path = Path(path).expanduser().resolve()
    base_method = _derive_method_name(resolved_path, method)
    if resolved_path.is_file():
        records = _read_jsonl(resolved_path)
    elif resolved_path.is_dir():
        records = _load_records_from_dir(resolved_path)
    else:
        raise FileNotFoundError(f"Evaluation input does not exist: {resolved_path}")

    normalized: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        row = dict(record)
        row["method"] = str(row.get("method") or base_method)
        row["input_source"] = str(resolved_path)
        row["sample_key"] = _sample_key(row, idx)
        row["generation_status"] = _default_generation_status(row)
        row["prompt_text"] = str(row.get("prompt_text") or row.get("text") or "")
        row["bbox_xyz"] = _coerce_bbox(row.get("bbox_xyz"))
        reference_mesh_path = row.get("reference_mesh_path") or row.get(
            "target_mesh_path"
        )
        row["reference_mesh_path"] = (
            None
            if reference_mesh_path in (None, "")
            else str(
                resolve_local_data_path(reference_mesh_path, anchor_path=resolved_path)
            )
        )
        row["generated_mesh_path"] = (
            None
            if row.get("generated_mesh_path") in (None, "")
            else str(
                resolve_local_data_path(row["generated_mesh_path"], anchor_path=resolved_path)
            )
        )
        row["sample_dir"] = (
            None
            if row.get("sample_dir") in (None, "")
            else str(resolve_local_data_path(row["sample_dir"], anchor_path=resolved_path))
        )
        normalized.append(row)
    return normalized


def build_default_output_dir(
    run_specs: list[tuple[Optional[str], Path]],
    requested_output_dir: Optional[str | Path],
) -> Path:
    if requested_output_dir is not None:
        return Path(requested_output_dir).expanduser().resolve()
    if len(run_specs) == 1 and run_specs[0][1].is_dir():
        return run_specs[0][1] / "evaluation"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (Path("./runs") / f"generation_evaluation_{timestamp}").resolve()


def _fscore_metric_suffix(pct: float) -> str:
    scaled = pct * 100.0
    if math.isclose(scaled, round(scaled), abs_tol=1e-9):
        return f"{int(round(scaled))}pct"
    return f"{scaled:.3f}".rstrip("0").rstrip(".").replace(".", "p") + "pct"


def _normalize_extent(vertices: np.ndarray) -> list[float]:
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    return normalize_bbox(tuple(float(value) for value in extent))


def _sample_surface_points(
    mesh: Any,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    positive_area = face_areas > 0.0
    if not bool(np.any(positive_area)):
        raise ValueError("Mesh has no positive-area faces for surface sampling")

    face_indices = np.flatnonzero(positive_area)
    probabilities = face_areas[positive_area]
    probabilities = probabilities / probabilities.sum()
    sampled_face_indices = face_indices[
        rng.choice(face_indices.shape[0], size=num_samples, replace=True, p=probabilities)
    ]

    triangles = vertices[faces[sampled_face_indices]]
    random_u = np.sqrt(rng.random(num_samples, dtype=np.float64))
    random_v = rng.random(num_samples, dtype=np.float64)
    w0 = 1.0 - random_u
    w1 = random_u * (1.0 - random_v)
    w2 = random_u * random_v
    points = (
        triangles[:, 0] * w0[:, None]
        + triangles[:, 1] * w1[:, None]
        + triangles[:, 2] * w2[:, None]
    )

    normals = np.asarray(mesh.face_normals[sampled_face_indices], dtype=np.float64)
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_norms, 1e-12)
    return points.astype(np.float32), normals.astype(np.float32)


def _query_nearest(
    target_points: np.ndarray,
    query_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(target_points)
    try:
        distances, indices = tree.query(query_points, k=1, workers=-1)
    except TypeError:
        distances, indices = tree.query(query_points, k=1)
    return np.asarray(distances, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def compute_pairwise_mesh_metrics(
    *,
    reference_mesh: Any,
    generated_mesh: Any,
    bbox_xyz: Optional[list[float]],
    surface_samples: int,
    fscore_threshold_pcts: Iterable[float],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    reference_points, reference_normals = _sample_surface_points(
        reference_mesh,
        num_samples=surface_samples,
        rng=rng,
    )
    generated_points, generated_normals = _sample_surface_points(
        generated_mesh,
        num_samples=surface_samples,
        rng=rng,
    )

    generated_to_reference_dist, generated_to_reference_idx = _query_nearest(
        target_points=reference_points,
        query_points=generated_points,
    )
    reference_to_generated_dist, reference_to_generated_idx = _query_nearest(
        target_points=generated_points,
        query_points=reference_points,
    )

    generated_to_reference_normals = reference_normals[generated_to_reference_idx]
    reference_to_generated_normals = generated_normals[reference_to_generated_idx]
    generated_normal_alignment = np.abs(
        np.sum(generated_normals * generated_to_reference_normals, axis=1)
    )
    reference_normal_alignment = np.abs(
        np.sum(reference_normals * reference_to_generated_normals, axis=1)
    )

    reference_bbox_xyz = _normalize_extent(np.asarray(reference_mesh.vertices))
    target_bbox_xyz = reference_bbox_xyz if bbox_xyz is None else [float(v) for v in bbox_xyz]
    target_bbox_diagonal = float(np.linalg.norm(np.asarray(target_bbox_xyz, dtype=np.float32)))

    metrics: dict[str, Any] = {
        "cd_l1": float(
            0.5
            * (
                float(reference_to_generated_dist.mean())
                + float(generated_to_reference_dist.mean())
            )
        ),
        "normal_consistency": float(
            0.5
            * (
                float(generated_normal_alignment.mean())
                + float(reference_normal_alignment.mean())
            )
        ),
    }

    for pct in fscore_threshold_pcts:
        suffix = _fscore_metric_suffix(float(pct))
        tau = float(target_bbox_diagonal * float(pct))
        precision = float(np.mean(generated_to_reference_dist <= tau))
        recall = float(np.mean(reference_to_generated_dist <= tau))
        fscore = (
            0.0
            if precision + recall <= 0.0
            else float(2.0 * precision * recall / (precision + recall))
        )
        metrics[f"fscore_{suffix}"] = fscore
    return metrics


def _write_sample_evaluation_json(
    record: dict[str, Any],
    payload: dict[str, Any],
    overwrite: bool,
) -> None:
    sample_dir = record.get("sample_dir")
    if sample_dir in (None, ""):
        return
    path = Path(str(sample_dir)) / "evaluation.json"
    if path.exists() and not overwrite:
        return
    save_json(path, payload)


def _sample_artifact_dir(record: dict[str, Any], output_root_dir: Optional[Path]) -> Optional[Path]:
    sample_dir = record.get("sample_dir")
    if sample_dir not in (None, ""):
        return Path(str(sample_dir))
    if output_root_dir is None:
        return None
    method = str(record.get("method") or "unknown")
    sample_key = str(record.get("sample_key") or "sample")
    return output_root_dir / "sample_artifacts" / method / sample_key


def _image_paths_from_render_dir(render_dir: Path, nviews: int) -> list[Path]:
    return [render_dir / f"{view_idx:03d}_textured.png" for view_idx in range(nviews)]


class MultiViewCLIPScorer:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizerFast

        model_path = Path(model_name).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"CLIP model directory not found: {model_path}")
        self.model_name = str(model_path)
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(
            self.model_name,
            local_files_only=True,
        ).eval().to(self.device)
        try:
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            self.image_processor = None
            self.tokenizer = None
        except OSError:
            image_size = int(getattr(self.model.config.vision_config, "image_size", 224))
            self.processor = None
            self.image_processor = CLIPImageProcessor(
                size={"shortest_edge": image_size},
                crop_size={"height": image_size, "width": image_size},
            )
            self.tokenizer = CLIPTokenizerFast.from_pretrained(
                self.model_name,
                local_files_only=True,
            )

    def score_image_paths(
        self,
        prompt_text: str,
        image_paths: list[str | Path],
    ) -> dict[str, Any]:
        import torch
        from PIL import Image

        resolved_paths = [Path(path).expanduser().resolve() for path in image_paths]
        images = [Image.open(path).convert("RGB") for path in resolved_paths]
        if self.processor is not None:
            image_inputs = self.processor(images=images, return_tensors="pt")
            text_inputs = self.processor(
                text=[prompt_text],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
        else:
            image_inputs = self.image_processor(images=images, return_tensors="pt")
            text_inputs = self.tokenizer(
                text=[prompt_text],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
        image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

        with torch.inference_mode():
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
        if not torch.is_tensor(image_features):
            image_features = getattr(
                image_features,
                "image_embeds",
                getattr(image_features, "pooler_output", None),
            )
        if not torch.is_tensor(text_features):
            text_features = getattr(
                text_features,
                "text_embeds",
                getattr(text_features, "pooler_output", None),
            )
        if image_features is None or text_features is None:
            raise TypeError("CLIP model did not return tensor image/text features.")
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scores = (image_features @ text_features.T).squeeze(-1).detach().cpu().float().tolist()
        if isinstance(scores, float):
            scores = [scores]
        clipped_scores = [max(float(score), 0.0) * 100.0 for score in scores]
        return {
            "mv_clip_model_name": self.model_name,
            "mv_clip_num_views": int(len(scores)),
            "mv_clip_view_scores": [float(score) for score in scores],
            "mv_clip_score_mean": float(np.mean(scores)) if scores else 0.0,
            "mv_clip_score_max": float(np.max(scores)) if scores else 0.0,
            "mv_clip_score_min": float(np.min(scores)) if scores else 0.0,
            "mv_clip_score_std": float(np.std(scores)) if scores else 0.0,
            "clipscore_model_name": self.model_name,
            "clipscore_num_views": int(len(clipped_scores)),
            "clipscore_view_scores": clipped_scores,
            "clipscore_mean": float(np.mean(clipped_scores)) if clipped_scores else 0.0,
            "clipscore_max": float(np.max(clipped_scores)) if clipped_scores else 0.0,
            "clipscore_min": float(np.min(clipped_scores)) if clipped_scores else 0.0,
            "clipscore_std": float(np.std(clipped_scores)) if clipped_scores else 0.0,
            "clipscore_cosine_view_scores": [float(score) for score in scores],
            "clipscore_cosine_mean": float(np.mean(scores)) if scores else 0.0,
            "clipscore_cosine_max": float(np.max(scores)) if scores else 0.0,
            "clipscore_cosine_min": float(np.min(scores)) if scores else 0.0,
            "clipscore_cosine_std": float(np.std(scores)) if scores else 0.0,
        }


def render_multiview_images(
    mesh_path: str | Path,
    render_dir: str | Path,
    *,
    nviews: int,
    img_resolution: int,
    overwrite: bool = False,
    render_fn: Optional[Any] = None,
    backend: str = "auto",
) -> tuple[list[str], str]:
    if nviews <= 0:
        raise ValueError(f"nviews must be positive, got {nviews}")
    resolved_render_dir = Path(render_dir).expanduser().resolve()
    expected_paths = _image_paths_from_render_dir(resolved_render_dir, nviews)
    if (not overwrite) and all(path.exists() for path in expected_paths):
        return [str(path) for path in expected_paths], "cached"

    resolved_render_dir.mkdir(parents=True, exist_ok=True)
    normalized_backend = str(backend).lower()
    if normalized_backend not in {
        "auto",
        "blender",
        "matplotlib",
        "pytorch3d",
        "simple",
        "legacy_gray",
    }:
        raise ValueError(f"Unsupported render backend {backend!r}")

    def _render_with_legacy_gray() -> list[str]:
        from PIL import Image
        from scipy import ndimage

        mesh = load_scaled_mesh(str(mesh_path))
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            raise ValueError(f"Cannot render empty mesh: {mesh_path}")

        center = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
        vertices = vertices - center
        triangles = vertices[faces]
        centroids = triangles.mean(axis=1)
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(normal_norms, 1e-6)

        resolution = int(img_resolution)
        view_scale = 2.15
        base_gray = 95.0
        gray_span = 60.0
        light_dir = np.asarray([0.30, 0.60, 0.50], dtype=np.float32)
        light_dir = light_dir / max(float(np.linalg.norm(light_dir)), 1e-6)
        image_paths: list[str] = []

        for view_idx in range(nviews):
            theta = 2.0 * np.pi * float(view_idx) / float(nviews)
            forward = np.asarray([np.sin(theta), 0.0, np.cos(theta)], dtype=np.float32)
            right = np.asarray([np.cos(theta), 0.0, -np.sin(theta)], dtype=np.float32)
            up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

            x = centroids @ right
            y = centroids @ up
            z = centroids @ forward
            px = np.rint((x / view_scale + 0.5) * (resolution - 1)).astype(np.int32)
            py = np.rint((0.5 - y / view_scale) * (resolution - 1)).astype(np.int32)
            valid = (px >= 0) & (px < resolution) & (py >= 0) & (py < resolution)
            px = px[valid]
            py = py[valid]
            z = z[valid]
            visible_normals = normals[valid]

            facing = np.abs(visible_normals @ forward)
            diffuse = np.maximum(visible_normals @ light_dir, 0.0)
            gray = np.clip(
                base_gray + gray_span * (0.45 * facing + 0.55 * diffuse),
                91.0,
                195.0,
            ).astype(np.uint8)

            flat_indices = py * resolution + px
            order = np.argsort(z)
            image = np.full(resolution * resolution, 255, dtype=np.uint8)
            image[flat_indices[order]] = gray[order]
            image = image.reshape(resolution, resolution)

            object_mask = image < 250
            dilated = ndimage.grey_dilation(image, size=(3, 3))
            object_mask = ndimage.binary_dilation(object_mask, iterations=1)
            image = np.where(object_mask, np.minimum(dilated, 190), 255).astype(np.uint8)
            object_mask = ndimage.binary_closing(image < 250, iterations=1)
            filled = ndimage.grey_erosion(image, size=(3, 3))
            image = np.where(object_mask, filled, 255).astype(np.uint8)

            path = resolved_render_dir / f"{view_idx:03d}_textured.png"
            Image.fromarray(image, mode="L").convert("RGB").save(path)
            image_paths.append(str(path))
        return image_paths

    def _render_with_simple() -> list[str]:
        from PIL import Image, ImageDraw

        mesh = load_scaled_mesh(str(mesh_path))
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            raise ValueError(f"Cannot render empty mesh: {mesh_path}")

        center = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
        vertices = vertices - center
        scale = float(np.max(np.abs(vertices)))
        vertices = vertices / max(scale, 1e-6)

        max_faces = 50000
        if faces.shape[0] > max_faces:
            step = int(np.ceil(faces.shape[0] / max_faces))
            faces = faces[::step]

        light_dir = np.asarray([0.35, -0.45, 0.82], dtype=np.float32)
        light_dir = light_dir / max(float(np.linalg.norm(light_dir)), 1e-6)

        image_paths: list[str] = []
        resolution = int(img_resolution)
        margin = 0.82
        elev = np.deg2rad(18.0)
        cos_elev, sin_elev = np.cos(elev), np.sin(elev)
        rx = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, cos_elev, -sin_elev], [0.0, sin_elev, cos_elev]],
            dtype=np.float32,
        )
        for view_idx in range(nviews):
            azim = np.deg2rad(float(view_idx) * 360.0 / float(nviews))
            cos_azim, sin_azim = np.cos(azim), np.sin(azim)
            rz = np.asarray(
                [
                    [cos_azim, -sin_azim, 0.0],
                    [sin_azim, cos_azim, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            rotated = vertices @ (rz @ rx).T
            tri = rotated[faces]
            face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            normal_norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
            face_normals = face_normals / np.maximum(normal_norm, 1e-6)
            visible = face_normals[:, 2] > -0.08
            if not np.any(visible):
                visible = np.ones(faces.shape[0], dtype=bool)
            tri = tri[visible]
            face_normals = face_normals[visible]
            depths = tri[:, :, 2].mean(axis=1)
            order = np.argsort(depths)

            xy = tri[:, :, :2]
            px = (xy[..., 0] * margin + 1.0) * 0.5 * (resolution - 1)
            py = (1.0 - (xy[..., 1] * margin + 1.0) * 0.5) * (resolution - 1)
            polys = np.stack([px, py], axis=-1)
            shading = np.clip(face_normals @ light_dir, -1.0, 1.0)
            shading = 0.38 + 0.44 * (0.5 * (shading + 1.0))

            image = Image.new("RGB", (resolution, resolution), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            for face_idx in order:
                points = [(float(x), float(y)) for x, y in polys[face_idx]]
                gray = int(np.clip(255.0 * shading[face_idx], 70, 225))
                draw.polygon(points, fill=(gray, gray, gray))

            path = resolved_render_dir / f"{view_idx:03d}_textured.png"
            image.save(path)
            image_paths.append(str(path))
        return image_paths

    def _render_with_matplotlib() -> list[str]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        mesh = load_scaled_mesh(str(mesh_path))
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        triangles = vertices[faces]
        normals = np.asarray(mesh.face_normals, dtype=np.float32)
        light_dir = np.asarray([0.35, 0.45, 0.82], dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)
        shading = np.clip(normals @ light_dir, -1.0, 1.0)
        shading = 0.28 + 0.62 * (0.5 * (shading + 1.0))
        facecolors = np.stack(
            [0.70 * shading, 0.74 * shading, 0.80 * shading, np.ones_like(shading)],
            axis=1,
        )

        image_paths: list[str] = []
        figure_size = max(float(img_resolution) / 100.0, 1.0)
        for view_idx in range(nviews):
            fig = plt.figure(figsize=(figure_size, figure_size), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            collection = Poly3DCollection(
                triangles,
                facecolors=facecolors,
                linewidths=0.0,
                edgecolors="none",
                antialiased=False,
            )
            ax.add_collection3d(collection)
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.set_zlim(-1.0, 1.0)
            ax.set_box_aspect((1.0, 1.0, 1.0))
            if hasattr(ax, "set_proj_type"):
                ax.set_proj_type("ortho")
            ax.view_init(elev=18.0, azim=float(view_idx) * 360.0 / float(nviews))
            ax.set_axis_off()
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
            path = resolved_render_dir / f"{view_idx:03d}_textured.png"
            fig.savefig(
                path,
                dpi=100,
                facecolor="white",
                edgecolor="white",
                transparent=False,
                bbox_inches="tight",
                pad_inches=0.0,
            )
            plt.close(fig)
            image_paths.append(str(path))
        return image_paths

    def _render_with_pytorch3d() -> list[str]:
        import os
        import sys
        import torch
        from PIL import Image
        try:
            from pytorch3d.renderer import (
                BlendParams,
                DirectionalLights,
                FoVPerspectiveCameras,
                HardPhongShader,
                MeshRasterizer,
                RasterizationSettings,
                TexturesVertex,
                look_at_view_transform,
            )
            from pytorch3d.structures import Meshes
        except ModuleNotFoundError:
            fallback_site_packages = os.environ.get("BLOCK3D_PYTORCH3D_SITE_PACKAGES")
            if fallback_site_packages is None:
                repo_root = Path(__file__).resolve().parents[1]
                fallback_site_packages = str(
                    repo_root
                    / ".runtime"
                    / ".texture_envs"
                    / "flashtex"
                    / "lib"
                    / "python3.10"
                    / "site-packages"
                )
            if fallback_site_packages not in sys.path:
                sys.path.append(fallback_site_packages)
            from pytorch3d.renderer import (
                BlendParams,
                DirectionalLights,
                FoVPerspectiveCameras,
                HardPhongShader,
                MeshRasterizer,
                RasterizationSettings,
                TexturesVertex,
                look_at_view_transform,
            )
            from pytorch3d.structures import Meshes

        mesh = load_scaled_mesh(str(mesh_path))
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        verts = torch.from_numpy(vertices)
        faces_tensor = torch.from_numpy(faces)
        center = (verts.max(dim=0).values + verts.min(dim=0).values) * 0.5
        verts = verts - center
        scale = verts.abs().max().clamp_min(1e-6)
        verts = verts / scale
        colors = torch.full((1, verts.shape[0], 3), 0.72, dtype=torch.float32)
        render_mesh = Meshes(
            verts=[verts.to(device)],
            faces=[faces_tensor.to(device)],
            textures=TexturesVertex(colors.to(device)),
        )
        raster_settings = RasterizationSettings(
            image_size=int(img_resolution),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )
        lights = DirectionalLights(device=device, direction=[[-0.6, 0.4, 1.0]])
        image_paths: list[str] = []
        for view_idx in range(nviews):
            azim = float(view_idx) * 360.0 / float(nviews)
            R, T = look_at_view_transform(dist=2.0, elev=30.0, azim=azim, device=device)
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=40.0)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(meshes_world=render_mesh)
            shader = HardPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=(1, 1, 1)),
            )
            image = shader(fragments, render_mesh)[0, ..., :3].detach().cpu().numpy()
            path = resolved_render_dir / f"{view_idx:03d}_textured.png"
            Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)
            image_paths.append(str(path))
        return image_paths

    if normalized_backend == "auto":
        backend_attempts = ["pytorch3d", "matplotlib", "legacy_gray", "simple"]
        if render_fn is not None:
            backend_attempts.insert(1, "blender")
    else:
        backend_attempts = [normalized_backend]
    errors: list[str] = []
    for backend_name in backend_attempts:
        try:
            if backend_name == "blender":
                if render_fn is None:
                    raise RuntimeError(
                        "The blender backend requires an explicit render_fn in this "
                        "Block3D package."
                    )
                actual_render_fn = render_fn
                image_paths = actual_render_fn(
                    str(Path(mesh_path).expanduser().resolve()),
                    str(resolved_render_dir),
                    nviews=int(nviews),
                    img_resolution=int(img_resolution),
                )
            elif backend_name == "pytorch3d":
                image_paths = _render_with_pytorch3d()
            elif backend_name == "simple":
                image_paths = _render_with_simple()
            elif backend_name == "legacy_gray":
                image_paths = _render_with_legacy_gray()
            else:
                image_paths = _render_with_matplotlib()
            return [str(Path(path).expanduser().resolve()) for path in image_paths], backend_name
        except Exception as error:
            errors.append(f"{backend_name}: {error!r}")
    raise RuntimeError(
        "Failed to render multi-view images. "
        + "; ".join(errors)
    )


def augment_records_with_multiview_clip(
    records: list[dict[str, Any]],
    *,
    scorer: Any,
    output_root_dir: Optional[str | Path],
    render_nviews: int,
    render_resolution: int,
    render_backend: str = "auto",
    overwrite_renders: bool = False,
    overwrite_sample_json: bool = False,
    render_fn: Optional[Any] = None,
) -> list[dict[str, Any]]:
    artifacts_root = (
        None
        if output_root_dir is None
        else Path(output_root_dir).expanduser().resolve()
    )
    augmented: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        for key in CLIP_RECORD_KEYS:
            updated.pop(key, None)
        updated["mv_clip_status"] = "skipped"
        updated["mv_clip_error"] = None
        generated_mesh_path = updated.get("generated_mesh_path")
        prompt_text = str(updated.get("prompt_text") or "")
        if not bool(updated.get("generation_success")):
            augmented.append(updated)
            _write_sample_evaluation_json(updated, updated, overwrite=overwrite_sample_json)
            continue
        if generated_mesh_path in (None, ""):
            updated["mv_clip_status"] = "error"
            updated["mv_clip_error"] = "missing generated_mesh_path"
            augmented.append(updated)
            _write_sample_evaluation_json(updated, updated, overwrite=overwrite_sample_json)
            continue
        if not prompt_text:
            updated["mv_clip_status"] = "error"
            updated["mv_clip_error"] = "missing prompt_text"
            augmented.append(updated)
            _write_sample_evaluation_json(updated, updated, overwrite=overwrite_sample_json)
            continue

        artifact_dir = _sample_artifact_dir(updated, artifacts_root)
        if artifact_dir is None:
            updated["mv_clip_status"] = "error"
            updated["mv_clip_error"] = "no artifact directory available for renders"
            augmented.append(updated)
            _write_sample_evaluation_json(updated, updated, overwrite=overwrite_sample_json)
            continue

        render_dir = artifact_dir / "renders" / "generated_mvclip"
        try:
            image_paths, used_backend = render_multiview_images(
                generated_mesh_path,
                render_dir,
                nviews=render_nviews,
                img_resolution=render_resolution,
                overwrite=overwrite_renders,
                render_fn=render_fn,
                backend=render_backend,
            )
            updated["mv_clip_status"] = "ok"
            updated["mv_clip_render_backend"] = used_backend
            updated["mv_clip_render_dir"] = str(render_dir)
            updated["mv_clip_image_paths"] = [str(path) for path in image_paths]
            updated.update(
                scorer.score_image_paths(
                    prompt_text=prompt_text,
                    image_paths=image_paths,
                )
            )
        except Exception as error:
            updated["mv_clip_status"] = "error"
            updated["mv_clip_error"] = repr(error)

        augmented.append(updated)
        _write_sample_evaluation_json(updated, updated, overwrite=overwrite_sample_json)
    return augmented


def evaluate_record(
    record: dict[str, Any],
    *,
    surface_samples: int,
    fscore_threshold_pcts: Iterable[float],
    seed: int,
    overwrite_sample_json: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = dict(record)
    output["evaluation_version"] = EVALUATION_VERSION
    output["surface_samples"] = int(surface_samples)
    output["fscore_threshold_pcts"] = [float(value) for value in fscore_threshold_pcts]
    output["generation_success"] = bool(
        str(record.get("generation_status", "")).lower() not in {"error", "failed", "missing"}
        and record.get("generated_mesh_path") not in (None, "")
    )
    output["valid_mesh"] = False
    output["evaluation_status"] = "skipped"
    output["evaluation_error"] = None

    if not output["generation_success"]:
        _write_sample_evaluation_json(record, output, overwrite=overwrite_sample_json)
        return output

    reference_mesh_path = record.get("reference_mesh_path")
    generated_mesh_path = record.get("generated_mesh_path")
    if reference_mesh_path in (None, ""):
        output["evaluation_status"] = "error"
        output["evaluation_error"] = "missing reference_mesh_path"
        _write_sample_evaluation_json(record, output, overwrite=overwrite_sample_json)
        return output
    if generated_mesh_path in (None, ""):
        output["evaluation_status"] = "error"
        output["evaluation_error"] = "missing generated_mesh_path"
        _write_sample_evaluation_json(record, output, overwrite=overwrite_sample_json)
        return output

    try:
        reference_mesh = load_scaled_mesh(str(reference_mesh_path))
        generated_mesh = load_scaled_mesh(str(generated_mesh_path))
        output["valid_mesh"] = True
        output["evaluation_status"] = "ok"
        output.update(
            compute_pairwise_mesh_metrics(
                reference_mesh=reference_mesh,
                generated_mesh=generated_mesh,
                bbox_xyz=record.get("bbox_xyz"),
                surface_samples=surface_samples,
                fscore_threshold_pcts=fscore_threshold_pcts,
                seed=seed,
            )
        )
    except Exception as error:
        output["evaluation_status"] = "error"
        output["evaluation_error"] = repr(error)

    _write_sample_evaluation_json(record, output, overwrite=overwrite_sample_json)
    return output


def evaluate_records(
    records: list[dict[str, Any]],
    *,
    surface_samples: int,
    fscore_threshold_pcts: Iterable[float] = DEFAULT_FSCORE_THRESHOLD_PCTS,
    seed: int = 0,
    overwrite_sample_json: bool = False,
) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    thresholds = tuple(float(value) for value in fscore_threshold_pcts)
    for idx, record in enumerate(records):
        evaluated.append(
            evaluate_record(
                record,
                surface_samples=surface_samples,
                fscore_threshold_pcts=thresholds,
                seed=int(seed) + idx,
                overwrite_sample_json=overwrite_sample_json,
            )
        )
    return evaluated


def summarize_evaluation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_records = len(records)
    generation_success_count = sum(1 for record in records if bool(record.get("generation_success")))
    evaluation_success_records = [
        record for record in records if str(record.get("evaluation_status")) == "ok"
    ]
    evaluation_success_count = len(evaluation_success_records)
    mv_clip_records = [record for record in records if "mv_clip_status" in record]
    mv_clip_success_count = sum(
        1 for record in mv_clip_records if str(record.get("mv_clip_status")) == "ok"
    )

    geometry_numeric_keys = list(NUMERIC_METRIC_KEYS)
    for record in records:
        for key, value in record.items():
            if key in geometry_numeric_keys:
                continue
            if key.startswith("fscore_") and isinstance(value, (int, float)):
                geometry_numeric_keys.append(key)

    metrics_summary: dict[str, Any] = {}
    for key in geometry_numeric_keys:
        values = [
            float(record[key])
            for record in evaluation_success_records
            if isinstance(record.get(key), (int, float, np.floating))
        ]
        if not values:
            continue
        metrics_summary[key] = summarize_numeric_series(values)
    mv_clip_success_records = [
        record for record in mv_clip_records if str(record.get("mv_clip_status")) == "ok"
    ]
    for key in CLIP_NUMERIC_METRIC_KEYS:
        values = [
            float(record[key])
            for record in mv_clip_success_records
            if isinstance(record.get(key), (int, float, np.floating))
        ]
        if not values:
            continue
        metrics_summary[key] = summarize_numeric_series(values)

    summary: dict[str, Any] = {
        "evaluation_version": EVALUATION_VERSION,
        "num_records": total_records,
        "generation_success_count": generation_success_count,
        "generation_success_rate": (
            float(generation_success_count) / float(total_records)
            if total_records > 0
            else 0.0
        ),
        "evaluation_success_count": evaluation_success_count,
        "evaluation_success_rate": (
            float(evaluation_success_count) / float(total_records)
            if total_records > 0
            else 0.0
        ),
        "metrics": metrics_summary,
    }
    if mv_clip_records:
        summary["mv_clip_count"] = len(mv_clip_records)
        summary["mv_clip_success_count"] = mv_clip_success_count
        summary["mv_clip_success_rate"] = (
            float(mv_clip_success_count) / float(len(mv_clip_records))
            if mv_clip_records
            else 0.0
        )
    return summary


def summarize_evaluation_by_method(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        method = str(record.get("method") or "unknown")
        grouped.setdefault(method, []).append(record)

    summaries: list[dict[str, Any]] = []
    for method in sorted(grouped):
        summary = summarize_evaluation_records(grouped[method])
        summary["method"] = method
        summaries.append(summary)
    return summaries


def build_evaluation_summary(
    *,
    run_specs: list[tuple[Optional[str], Path]],
    records: list[dict[str, Any]],
    surface_samples: int,
    fscore_threshold_pcts: Iterable[float],
) -> dict[str, Any]:
    overall = summarize_evaluation_records(records)
    return {
        "evaluation_version": EVALUATION_VERSION,
        "surface_samples": int(surface_samples),
        "fscore_threshold_pcts": [float(value) for value in fscore_threshold_pcts],
        "input_runs": [
            {
                "method": _derive_method_name(path, method),
                "path": str(path),
            }
            for method, path in run_specs
        ],
        "overall": overall,
        "by_method": summarize_evaluation_by_method(records),
    }
