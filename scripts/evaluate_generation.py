import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from block3d.evaluation import (
    DEFAULT_FSCORE_THRESHOLD_PCTS,
    MultiViewCLIPScorer,
    augment_records_with_multiview_clip,
    build_default_output_dir,
    build_evaluation_summary,
    evaluate_records,
    load_evaluation_records,
    parse_run_spec,
    save_csv,
    save_json,
    save_jsonl,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated meshes with CD, F-Score, and Normal Consistency."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Input in the form method=PATH or PATH. PATH can be a sample-eval directory "
            "containing samples.jsonl/summary.json or a jsonl file with evaluation records."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for evaluation outputs. Defaults to <run>/evaluation for a single "
            "directory input, otherwise a timestamped directory under ./runs."
        ),
    )
    parser.add_argument(
        "--surface-samples",
        type=int,
        default=8192,
        help="Number of surface points sampled per mesh for geometry metrics.",
    )
    parser.add_argument(
        "--fscore-threshold-pct",
        type=float,
        action="append",
        default=None,
        help=(
            "Relative F-score threshold as a fraction of the target bbox diagonal. "
            "Can be repeated. Defaults to 0.01 and 0.02."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed for deterministic surface sampling.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the total number of records evaluated after loading inputs.",
    )
    parser.add_argument(
        "--overwrite-sample-json",
        default=False,
        action="store_true",
        help="Rewrite per-sample evaluation.json even if it already exists.",
    )
    parser.add_argument(
        "--skip-geometry",
        default=False,
        action="store_true",
        help="Skip CD/F-score/normal metrics and only run requested auxiliary metrics.",
    )
    parser.add_argument(
        "--clipscore",
        default=False,
        action="store_true",
        help="Compute CLIPScore from gray multi-view renders of each generated mesh.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="model_weights/clip-vit-large-patch14",
        help="Local CLIP model directory used for CLIPScore.",
    )
    parser.add_argument(
        "--clip-device",
        type=str,
        default="cuda",
        help="Device for CLIPScore model. Use cpu if CUDA is unavailable.",
    )
    parser.add_argument(
        "--clip-render-nviews",
        type=int,
        default=8,
        help="Number of gray mesh views rendered per sample for CLIPScore.",
    )
    parser.add_argument(
        "--clip-render-resolution",
        type=int,
        default=512,
        help="Resolution of gray mesh renders used for CLIPScore.",
    )
    parser.add_argument(
        "--clip-render-backend",
        choices=("auto", "pytorch3d", "matplotlib", "legacy_gray", "simple"),
        default="auto",
        help="Renderer backend for CLIPScore gray multi-view images.",
    )
    parser.add_argument(
        "--overwrite-clip-renders",
        default=False,
        action="store_true",
        help="Re-render CLIPScore gray multi-view images even if cached renders exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_specs = [parse_run_spec(spec) for spec in args.run]
    output_dir = build_default_output_dir(run_specs, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fscore_threshold_pcts = (
        DEFAULT_FSCORE_THRESHOLD_PCTS
        if args.fscore_threshold_pct is None
        else tuple(float(value) for value in args.fscore_threshold_pct)
    )

    records = []
    for method, path in run_specs:
        records.extend(load_evaluation_records(path, method=method))
    if args.max_samples is not None:
        records = records[: max(int(args.max_samples), 0)]

    if args.skip_geometry:
        evaluated_records = [dict(record) for record in records]
        for record in evaluated_records:
            record["generation_success"] = bool(
                str(record.get("generation_status", "")).lower()
                not in {"error", "failed", "missing"}
                and record.get("generated_mesh_path") not in (None, "")
            )
            record["evaluation_version"] = 2
            record["surface_samples"] = int(args.surface_samples)
            record["fscore_threshold_pcts"] = [float(value) for value in fscore_threshold_pcts]
            record["valid_mesh"] = None
            record["evaluation_status"] = "skipped"
            record["evaluation_error"] = None
    else:
        evaluated_records = evaluate_records(
            records,
            surface_samples=args.surface_samples,
            fscore_threshold_pcts=fscore_threshold_pcts,
            seed=args.seed,
            overwrite_sample_json=args.overwrite_sample_json,
        )
    if args.clipscore:
        scorer = MultiViewCLIPScorer(args.clip_model, device=args.clip_device)
        evaluated_records = augment_records_with_multiview_clip(
            evaluated_records,
            scorer=scorer,
            output_root_dir=output_dir,
            render_nviews=args.clip_render_nviews,
            render_resolution=args.clip_render_resolution,
            render_backend=args.clip_render_backend,
            overwrite_renders=args.overwrite_clip_renders,
            overwrite_sample_json=args.overwrite_sample_json,
        )
    summary = build_evaluation_summary(
        run_specs=run_specs,
        records=evaluated_records,
        surface_samples=args.surface_samples,
        fscore_threshold_pcts=fscore_threshold_pcts,
    )

    save_jsonl(output_dir / "samples.jsonl", evaluated_records)
    save_csv(output_dir / "samples.csv", evaluated_records)
    save_json(output_dir / "summary.json", summary)

    print(f"Saved {len(evaluated_records)} evaluated records to {output_dir / 'samples.jsonl'}")
    print(f"Saved evaluation summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
