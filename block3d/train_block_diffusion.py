import argparse
from dataclasses import dataclass
from datetime import timedelta
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from block3d.benchmarking import load_benchmark_prompts
from block3d.training import (
    ObjaverseDataset,
    SampleEvalSpec,
    collate_objaverse_batch,
    discover_objaverse_entries,
    prepare_sample_eval_specs,
    split_objaverse_entries,
)


DEFAULT_TRAIN_CONFIG_PATH = "block3d/configs/train_block3d.yaml"


def _load_block_diffusion_runtime() -> tuple[type[Any], Any, Any]:
    from block3d.training import Block3DTrainer
    from block3d.training.block_diffusion import peek_training_state, select_device

    return Block3DTrainer, peek_training_state, select_device


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    backend: str | None = None
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


class ResumeDistributedSampler(DistributedSampler):
    def __init__(
        self,
        *args: Any,
        resume_epoch: int,
        resume_batch_idx: int,
        batch_size: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.resume_epoch = int(resume_epoch)
        self.resume_batch_idx = max(int(resume_batch_idx), 0)
        self.resume_batch_size = max(int(batch_size), 1)
        self.handles_resume_batch_idx = self.resume_batch_idx > 0

    def _resume_skip_count(self) -> int:
        if self.epoch != self.resume_epoch or self.resume_batch_idx <= 0:
            return 0
        return min(super().__len__(), self.resume_batch_idx * self.resume_batch_size)

    def __iter__(self):
        indices = list(super().__iter__())
        skip_count = min(len(indices), self._resume_skip_count())
        if skip_count > 0:
            indices = indices[skip_count:]
        return iter(indices)

    def __len__(self) -> int:
        return max(0, super().__len__() - self._resume_skip_count())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Block3D with block diffusion."
    )
    parser.add_argument(
        "--train-config-path",
        type=str,
        default=DEFAULT_TRAIN_CONFIG_PATH,
        help="Path to the training hyperparameter config YAML. CLI flags override config values.",
    )
    return parser
    

def _load_training_config_defaults(
    parser: argparse.ArgumentParser,
    train_config_path: str | None,
) -> dict[str, Any]:
    if train_config_path is None:
        return {}

    config_path = Path(train_config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    payload = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Training config at {config_path} must be a mapping")

    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(payload) - valid_keys)
    if unknown_keys:
        raise ValueError(
            f"Training config {config_path} contains unknown keys: {unknown_keys}"
        )

    return dict(payload)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--train-config-path",
        type=str,
        default=DEFAULT_TRAIN_CONFIG_PATH,
    )
    pre_args, _ = pre_parser.parse_known_args()
    parser = _build_parser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory of the local Objaverse-style dataset.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional train manifest (.json/.jsonl/.csv/.tsv) describing mesh/text pairs.",
    )
    parser.add_argument(
        "--val-manifest-path",
        type=str,
        default=None,
        help="Optional validation manifest. If omitted, val split is taken from the train entries.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Validation split ratio used only when --val-manifest-path is not provided.",
    )
    parser.add_argument(
        "--skip-manifest-path-verification",
        default=False,
        action="store_true",
        help=(
            "Skip per-record mesh/pair existence checks when loading train/val manifests. "
            "Useful for pair-only training where most mesh_path values are traceability-only "
            "and do not exist locally."
        ),
    )
    parser.add_argument(
        "--skip-manifest-snapshot",
        default=False,
        action="store_true",
        help=(
            "Do not rewrite discovered train/val entries into the run directory. "
            "This avoids long rank-0 startup I/O on very large manifests; the source "
            "manifest paths are still recorded in the resolved config and dataset summary."
        ),
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="block3d/configs/block3d.yaml",
        help="Path to the model config.",
    )
    parser.add_argument(
        "--gpt-ckpt-path",
        type=str,
        default=None,
        help="GPT initialization checkpoint. Required unless resuming or --train-from-scratch is set.",
    )
    parser.add_argument(
        "--shape-ckpt-path",
        type=str,
        default=None,
        help="Checkpoint for Block3D's shape autoencoder. Required unless provided by a resume state.",
    )
    parser.add_argument(
        "--resume-trainer-state",
        type=str,
        default=None,
        help="Path to a saved trainer state for resuming optimizer/scheduler/global_step.",
    )
    parser.add_argument(
        "--train-from-scratch",
        default=False,
        action="store_true",
        help="Allow training without a GPT initialization checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory used for checkpoints and logs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Legacy epoch-based training budget. Prefer --max-steps for LLM-style training.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Primary training budget in optimizer steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Global gradient clipping norm. Use 0 to disable.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of microbatches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Warmup optimizer steps for the cosine scheduler.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Minimum learning rate as a fraction of the base learning rate for cosine decay.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Optional EMA decay in (0, 1). Set 0 to disable EMA.",
    )
    parser.add_argument(
        "--save-model-only-interval",
        "--save-interval",
        dest="save_model_only_interval",
        type=int,
        default=0,
        help="Save model-only checkpoints every N optimizer steps.",
    )
    parser.add_argument(
        "--save-full-state-interval",
        type=int,
        default=0,
        help="Save resumable trainer state every N optimizer steps.",
    )
    parser.add_argument(
        "--save-final-trainer-state",
        default=False,
        action="store_true",
        help="Save trainer_final.pt at the end of training. Disabled by default because it is large.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log metrics every N optimizer steps.",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=0,
        help="Run validation every N optimizer steps. Use 0 to disable periodic validation.",
    )
    parser.add_argument(
        "--val-max-batches",
        type=int,
        default=None,
        help="Optional cap on validation batches per validation run.",
    )
    parser.add_argument(
        "--sample-eval-interval",
        type=int,
        default=0,
        help="Run periodic sample generation eval every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--sample-eval-max-samples",
        type=int,
        default=0,
        help="Maximum number of prompts to generate per sample eval run.",
    )
    parser.add_argument(
        "--sample-eval-manifest-path",
        type=str,
        default=None,
        help=(
            "Optional prompt+bbox+mesh manifest used as the fixed sample-eval set. "
            "When provided, sample eval specs come from this manifest instead of "
            "being sampled from the train/validation pool."
        ),
    )
    parser.add_argument(
        "--sample-eval-selection-seed",
        type=int,
        default=None,
        help="Optional seed used to choose a fixed random subset of sample-eval objects.",
    )
    parser.add_argument(
        "--sample-eval-resample-each-interval",
        default=False,
        action="store_true",
        help="Resample visualization prompts from the validation pool at every sample-eval interval.",
    )
    parser.add_argument(
        "--sample-eval-resolution-base",
        type=float,
        default=8.0,
        help="Resolution base used when decoding sample eval meshes.",
    )
    parser.add_argument(
        "--sample-eval-chunk-size",
        type=int,
        default=100_000,
        help="Chunk size used during sample eval shape decoding.",
    )
    parser.add_argument(
        "--sample-eval-top-p",
        type=float,
        default=None,
        help="Optional top-p sampling used during sample eval generation.",
    )
    parser.add_argument(
        "--sample-eval-guidance-scale",
        type=float,
        default=None,
        help="Optional CFG guidance scale used during sample eval generation.",
    )
    parser.add_argument(
        "--sample-eval-num-diffusion-steps",
        type=int,
        default=None,
        help="Optional override for block diffusion denoising steps during sample eval.",
    )
    parser.add_argument(
        "--sample-eval-sampling-strategy",
        type=str,
        default="block3d",
        choices=("block3d",),
        help="Sampling strategy used during sample eval block diffusion generation.",
    )
    parser.add_argument(
        "--sample-eval-render-multiview",
        default=False,
        action="store_true",
        help="Render multi-view images for each generated sample-eval mesh.",
    )
    parser.add_argument(
        "--sample-eval-render-nviews",
        type=int,
        default=4,
        help="Number of rendered views per sample-eval mesh when rendering is enabled.",
    )
    parser.add_argument(
        "--sample-eval-render-resolution",
        type=int,
        default=256,
        help="Per-view image resolution for sample-eval multi-view rendering.",
    )
    parser.add_argument(
        "--sample-eval-render-backend",
        type=str,
        default="auto",
        choices=("auto", "matplotlib"),
        help="Rendering backend used for sample-eval multi-view visualization.",
    )
    parser.add_argument(
        "--geometry-eval-interval",
        type=int,
        default=0,
        help="Run full geometry evaluation every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--geometry-eval-max-samples",
        type=int,
        default=None,
        help="Number of random validation objects sampled per geometry eval. Omit to use the full random pool.",
    )
    parser.add_argument(
        "--geometry-eval-fixed-max-samples",
        type=int,
        default=0,
        help="Number of fixed validation objects included in every geometry eval run.",
    )
    parser.add_argument(
        "--geometry-eval-selection-seed",
        type=int,
        default=None,
        help="Optional seed used for the random geometry-eval pool and its interval resampling.",
    )
    parser.add_argument(
        "--geometry-eval-fixed-selection-seed",
        type=int,
        default=None,
        help="Optional seed used to choose the fixed geometry-eval objects. Defaults to --seed when omitted.",
    )
    parser.add_argument(
        "--geometry-eval-fixed-manifest-path",
        type=str,
        default=None,
        help=(
            "Optional prompt+bbox+mesh manifest used as the fixed geometry-eval set. "
            "When provided, the fixed geometry-eval specs come from this manifest instead "
            "of being sampled from the validation pool. Bboxes normalize metric thresholds "
            "and are not passed to the generator."
        ),
    )
    parser.add_argument(
        "--geometry-eval-resample-each-interval",
        default=False,
        action="store_true",
        help="Resample geometry-eval objects from the validation pool at every interval.",
    )
    parser.add_argument(
        "--geometry-eval-surface-samples",
        type=int,
        default=8192,
        help="Number of surface samples used per mesh during geometry evaluation.",
    )
    parser.add_argument(
        "--geometry-eval-save-generated-meshes",
        default=False,
        action="store_true",
        help="Persist generated meshes and tokens for geometry eval runs.",
    )
    parser.add_argument(
        "--point-samples",
        type=int,
        default=8192,
        help="Number of surface points sampled per mesh.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for sampled point clouds and bbox data.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on the number of train samples after any split.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap on the number of validation samples.",
    )
    parser.add_argument(
        "--train-t-min",
        type=float,
        default=None,
        help="Minimum per-block masking rate during training.",
    )
    parser.add_argument(
        "--train-t-max",
        type=float,
        default=None,
        help="Maximum per-block masking rate during training.",
    )
    parser.add_argument(
        "--training-strategy",
        type=str,
        default="block3d",
        choices=("block3d",),
        help="Training corruption strategy used to construct block diffusion targets.",
    )
    parser.add_argument(
        "--cfg-dropout-prob",
        type=float,
        default=None,
        help="Optional classifier-free condition dropout probability used during training.",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16"),
        help="Autocast dtype used on CUDA.",
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="float32",
        choices=("float32", "bfloat16", "float16"),
        help="Parameter/storage dtype used when materializing models.",
    )
    parser.add_argument(
        "--disable-grad-scaler",
        default=False,
        action="store_true",
        help="Disable GradScaler even when using float16 autocast.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        dest="activation_checkpointing",
        default=None,
        action="store_true",
        help="Enable GPT activation checkpointing to trade compute for lower memory.",
    )
    parser.add_argument(
        "--disable-activation-checkpointing",
        dest="activation_checkpointing",
        action="store_false",
        help="Disable GPT activation checkpointing.",
    )
    parser.add_argument(
        "--offload-shape-model-to-cpu",
        dest="offload_shape_model_to_cpu",
        default=None,
        action="store_true",
        help="Keep the frozen shape tokenizer on CPU except when encode/decode is needed.",
    )
    parser.add_argument(
        "--disable-offload-shape-model-to-cpu",
        dest="offload_shape_model_to_cpu",
        action="store_false",
        help="Keep the frozen shape tokenizer resident on the training device.",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        default=True,
        action="store_true",
        help="Enable DDP unused-parameter detection. Recommended for the current block-diffusion training path.",
    )
    parser.add_argument(
        "--no-ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
        help="Disable DDP unused-parameter detection for maximum throughput when every parameter participates in loss each step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional explicit device, e.g. cuda, cpu, mps, cuda:0.",
    )
    parser.add_argument(
        "--ddp-backend",
        type=str,
        default="nccl",
        choices=("nccl", "gloo"),
        help="torch.distributed backend used under torchrun.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="Optional TensorBoard log directory. Defaults to <output-dir>/logs/tensorboard.",
    )
    parser.add_argument(
        "--disable-tensorboard",
        default=False,
        action="store_true",
        help="Disable TensorBoard event logging.",
    )
    parser.add_argument(
        "--tensorboard-flush-secs",
        type=int,
        default=30,
        help="How often TensorBoard SummaryWriter flushes event files.",
    )
    config_defaults = _load_training_config_defaults(parser, pre_args.train_config_path)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _resolve_dataloader_multiprocessing_context(
    *,
    device: torch.device,
    num_workers: int,
) -> str | None:
    # Trainer initialization loads CUDA state and tokenizer caches before DataLoader
    # workers are created, so CUDA-unsafe fork workers are not safe here.
    if num_workers <= 0:
        return None
    if device.type == "cuda":
        return "spawn"
    return None


def _truncate_entries(entries: list[dict], max_samples: int | None) -> list[dict]:
    if max_samples is None:
        return entries
    return entries[: max(max_samples, 0)]


def _serialize_entry(entry: dict[str, Any]) -> dict[str, str]:
    serialized: dict[str, Any] = {}
    for key, value in entry.items():
        if isinstance(value, Path):
            serialized[key] = str(value.expanduser().resolve())
        else:
            serialized[key] = value

    serialized["mesh_path"] = str(Path(entry["mesh_path"]).expanduser().resolve())
    serialized["text"] = str(entry["text"])
    return serialized


def _write_entries_manifest(path: Path, entries: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for entry in entries:
            handle.write(json.dumps(_serialize_entry(entry), sort_keys=True) + "\n")
    return path


def _sample_eval_specs_path(output_dir: Path) -> Path:
    return output_dir / "manifests" / "sample_eval_specs.json"


def _geometry_eval_specs_path(output_dir: Path) -> Path:
    return output_dir / "manifests" / "geometry_eval_specs.json"


def _geometry_eval_fixed_specs_path(output_dir: Path) -> Path:
    return output_dir / "manifests" / "geometry_eval_fixed_specs.json"


def _save_sample_eval_specs(path: Path, specs: list[SampleEvalSpec]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True)
    )
    return path


def _load_sample_eval_specs(path: Path) -> list[SampleEvalSpec]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected list of sample eval specs in {path}")
    return [SampleEvalSpec(**spec) for spec in payload]


def _load_benchmark_sample_eval_specs(
    manifest_path: str | Path,
    *,
    max_samples: int | None = None,
) -> list[SampleEvalSpec]:
    prompts = load_benchmark_prompts(manifest_path)
    if max_samples is not None:
        prompts = prompts[: max(int(max_samples), 0)]
    return [
        SampleEvalSpec(
            prompt_text=str(prompt.prompt_text),
            bbox_xyz=[float(value) for value in prompt.bbox_xyz],
            mesh_path=str(prompt.mesh_path),
        )
        for prompt in prompts
    ]


def _resolve_sample_eval_specs(
    *,
    output_dir: Path,
    entries: list[dict],
    max_samples: int,
    benchmark_manifest_path: str | Path | None,
    selection_seed: int | None,
    resume_state_present: bool,
    bad_samples_path: Path,
) -> list[SampleEvalSpec]:
    specs_path = _sample_eval_specs_path(output_dir)
    if max_samples <= 0:
        _save_sample_eval_specs(specs_path, [])
        return []
    if not entries and benchmark_manifest_path is None:
        return []

    if resume_state_present and specs_path.exists():
        specs = _load_sample_eval_specs(specs_path)
        logging.info("Loaded %d persisted sample eval specs from %s", len(specs), specs_path)
        return specs

    if benchmark_manifest_path is not None:
        specs = _load_benchmark_sample_eval_specs(
            benchmark_manifest_path,
            max_samples=max_samples,
        )
    else:
        specs = prepare_sample_eval_specs(
            entries,
            max_samples=max_samples,
            bad_samples_path=str(bad_samples_path),
            selection_seed=selection_seed,
        )
    _save_sample_eval_specs(specs_path, specs)
    return specs


def _resolve_geometry_eval_specs(
    *,
    output_dir: Path,
    entries: list[dict],
    max_samples: int | None,
    selection_seed: int | None,
    resume_state_present: bool,
    bad_samples_path: Path,
) -> list[SampleEvalSpec]:
    specs_path = _geometry_eval_specs_path(output_dir)
    if not entries:
        return []
    if max_samples is not None and max_samples <= 0:
        _save_sample_eval_specs(specs_path, [])
        return []

    if resume_state_present and specs_path.exists():
        specs = _load_sample_eval_specs(specs_path)
        logging.info(
            "Loaded %d persisted geometry eval specs from %s",
            len(specs),
            specs_path,
        )
        return specs

    effective_max_samples = len(entries) if max_samples is None else int(max_samples)
    specs = prepare_sample_eval_specs(
        entries,
        max_samples=min(effective_max_samples, len(entries)),
        bad_samples_path=str(bad_samples_path),
        selection_seed=selection_seed,
    )
    _save_sample_eval_specs(specs_path, specs)
    return specs


def _resolve_geometry_eval_fixed_specs(
    *,
    output_dir: Path,
    entries: list[dict],
    max_samples: int,
    benchmark_manifest_path: str | Path | None,
    selection_seed: int | None,
    resume_state_present: bool,
    bad_samples_path: Path,
) -> list[SampleEvalSpec]:
    specs_path = _geometry_eval_fixed_specs_path(output_dir)
    if max_samples <= 0:
        _save_sample_eval_specs(specs_path, [])
        return []
    if not entries and benchmark_manifest_path is None:
        return []

    if resume_state_present and specs_path.exists():
        specs = _load_sample_eval_specs(specs_path)
        logging.info(
            "Loaded %d persisted fixed geometry eval specs from %s",
            len(specs),
            specs_path,
        )
        return specs

    if benchmark_manifest_path is not None:
        specs = _load_benchmark_sample_eval_specs(
            benchmark_manifest_path,
            max_samples=max_samples,
        )
    else:
        specs = prepare_sample_eval_specs(
            entries,
            max_samples=min(int(max_samples), len(entries)),
            bad_samples_path=str(bad_samples_path),
            selection_seed=selection_seed,
        )
    _save_sample_eval_specs(specs_path, specs)
    return specs


def _get_git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _infer_conda_environment() -> str | None:
    executable_parts = Path(sys.executable).parts
    if "envs" in executable_parts:
        envs_idx = executable_parts.index("envs")
        if envs_idx + 1 < len(executable_parts):
            return executable_parts[envs_idx + 1]
    return os.environ.get("CONDA_DEFAULT_ENV")


def _cpu_model_name() -> str | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return value
        except OSError:
            pass
    value = platform.processor().strip()
    return value or os.environ.get("PROCESSOR_IDENTIFIER")


def _system_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return page_size * page_count


def _collect_environment_metadata(device: torch.device, cwd: Path) -> dict[str, Any]:
    cuda_device_names: list[str] = []
    cuda_device_total_memory_bytes: list[int] = []
    if torch.cuda.is_available():
        cuda_device_names = [
            torch.cuda.get_device_name(device_idx)
            for device_idx in range(torch.cuda.device_count())
        ]
        cuda_device_total_memory_bytes = [
            int(torch.cuda.get_device_properties(device_idx).total_memory)
            for device_idx in range(torch.cuda.device_count())
        ]

    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_model": _cpu_model_name(),
        "cpu_logical_count": os.cpu_count(),
        "system_memory_bytes": _system_memory_bytes(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": cuda_device_names,
        "cuda_device_total_memory_bytes": cuda_device_total_memory_bytes,
        "selected_device": str(device),
        "command": sys.argv,
        "cwd": str(cwd),
        "git_commit": _get_git_commit(cwd),
        "conda_default_env": _infer_conda_environment(),
    }


def _resolve_resume_defaults(args: argparse.Namespace) -> dict | None:
    if args.resume_trainer_state is None:
        return None

    _, peek_training_state, _ = _load_block_diffusion_runtime()
    trainer_state = peek_training_state(args.resume_trainer_state)
    if args.gpt_ckpt_path is None:
        args.gpt_ckpt_path = trainer_state.get("model_checkpoint_path") or trainer_state.get(
            "gpt_ckpt_path"
        )
    if args.shape_ckpt_path is None:
        args.shape_ckpt_path = trainer_state.get("shape_ckpt_path")
    return trainer_state


def _validate_required_training_args(args: argparse.Namespace) -> None:
    if args.data_root is None:
        raise ValueError(
            "--data-root is required, either from CLI or --train-config-path."
        )
    if args.output_dir is None:
        raise ValueError(
            "--output-dir is required, either from CLI or --train-config-path."
        )
    if args.max_steps is None and args.epochs is None:
        raise ValueError(
            "A training budget is required. Set --max-steps in the training config or CLI. "
            "--epochs is only kept as a legacy fallback."
        )
    if args.max_steps is not None and int(args.max_steps) <= 0:
        raise ValueError(f"--max-steps must be positive, got {args.max_steps}")
    if args.epochs is not None and int(args.epochs) <= 0:
        raise ValueError(f"--epochs must be positive when provided, got {args.epochs}")


def _write_training_config_snapshot(
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    snapshot_path = output_dir / "resolved_train_config.yaml"
    payload = {
        key: value
        for key, value in vars(args).items()
        if key != "device" or value is not None
    }
    OmegaConf.save(config=OmegaConf.create(payload), f=str(snapshot_path))
    return snapshot_path


def _training_budget_summary(args: argparse.Namespace) -> dict[str, Any]:
    budget_mode = "steps" if args.max_steps is not None else "epochs"
    return {
        "budget_mode": budget_mode,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
    }


def _should_resume_fast_load_manifest(
    args: argparse.Namespace,
    resume_state: dict | None,
) -> bool:
    if resume_state is None:
        return False
    # Resume runs already operate on a frozen manifest snapshot. Re-validating every
    # path in a 100k+ scale manifest on every rank makes restart latency dominated by
    # startup I/O instead of actual training recovery.
    return args.manifest_path is not None


def _should_skip_manifest_path_verification(
    args: argparse.Namespace,
    resume_state: dict | None,
) -> bool:
    if bool(args.skip_manifest_path_verification):
        return True
    return _should_resume_fast_load_manifest(args, resume_state)


def _configure_logging(distributed: DistributedContext) -> None:
    logging.basicConfig(
        level=logging.INFO if distributed.is_main_process else logging.WARNING,
        format=f"%(asctime)s %(levelname)s [rank {distributed.rank}] %(message)s",
        force=True,
    )


def _setup_distributed(ddp_backend: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(enabled=False)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun multi-GPU training requires CUDA.")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=ddp_backend,
            timeout=timedelta(minutes=30),
        )
    return DistributedContext(
        enabled=True,
        backend=ddp_backend,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def _distributed_barrier(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def _cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    distributed = _setup_distributed(args.ddp_backend)
    _configure_logging(distributed)

    try:
        set_seed(args.seed + distributed.rank)
        _validate_required_training_args(args)
        if args.max_steps is not None and args.epochs is not None and distributed.is_main_process:
            logging.info(
                "Both max_steps=%s and epochs=%s are set. Training will stop by max_steps.",
                args.max_steps,
                args.epochs,
            )

        resume_state = _resolve_resume_defaults(args)
        if not args.train_from_scratch and args.gpt_ckpt_path is None:
            raise ValueError(
                "A GPT checkpoint is required unless --train-from-scratch is set or "
                "--resume-trainer-state provides a saved model_checkpoint_path."
            )
        if args.shape_ckpt_path is None:
            raise ValueError(
                "--shape-ckpt-path is required unless --resume-trainer-state provides it."
            )

        trainer_cls, _, select_device = _load_block_diffusion_runtime()
        if distributed.enabled:
            if args.device is not None and distributed.is_main_process:
                logging.info(
                    "--device=%s is ignored under torchrun; using cuda:%d for local rank.",
                    args.device,
                    distributed.local_rank,
                )
            device = torch.device("cuda", distributed.local_rank)
        elif args.device is not None:
            device = torch.device(args.device)
        else:
            device = select_device()
        logging.info("Using device: %s", device)

        trainer_init_started_at = time.perf_counter()
        logging.info(
            "Initializing trainer and loading model weights onto %s before dataset discovery.",
            device,
        )
        trainer = trainer_cls(
            config_path=args.config_path,
            gpt_ckpt_path=args.gpt_ckpt_path,
            shape_ckpt_path=args.shape_ckpt_path,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            train_t_min=args.train_t_min,
            train_t_max=args.train_t_max,
            training_strategy=args.training_strategy,
            cfg_dropout_prob=args.cfg_dropout_prob,
            grad_clip_norm=args.grad_clip_norm,
            grad_accum_steps=args.grad_accum_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            ema_decay=args.ema_decay,
            amp_dtype=args.amp_dtype,
            model_dtype=args.model_dtype,
            use_grad_scaler=not args.disable_grad_scaler,
            activation_checkpointing=bool(args.activation_checkpointing),
            offload_shape_model_to_cpu=bool(args.offload_shape_model_to_cpu),
            ddp_find_unused_parameters=bool(args.ddp_find_unused_parameters),
            distributed_rank=distributed.rank,
            distributed_local_rank=distributed.local_rank,
            distributed_world_size=distributed.world_size,
        )
        logging.info(
            "Trainer initialization finished in %.2fs; GPU weights are loaded before dataset preparation.",
            time.perf_counter() - trainer_init_started_at,
        )

        resume_fast_load_manifest = _should_resume_fast_load_manifest(args, resume_state)
        skip_manifest_path_verification = _should_skip_manifest_path_verification(
            args,
            resume_state,
        )
        if skip_manifest_path_verification and distributed.is_main_process:
            if resume_fast_load_manifest:
                logging.info(
                    "Resume fast path enabled: loading manifest entries without per-record path "
                    "verification before training resumes."
                )
            else:
                logging.info(
                    "Skipping manifest path verification for train/val manifests. "
                    "This is intended for pair-only training where local mesh_path values "
                    "may be missing."
                )

        discovered_entries, discovered_summary = discover_objaverse_entries(
            root=args.data_root,
            manifest_path=args.manifest_path,
            verify_paths=not skip_manifest_path_verification,
        )
        if args.val_manifest_path is not None:
            train_entries = discovered_entries
            val_entries, val_summary = discover_objaverse_entries(
                root=args.data_root,
                manifest_path=args.val_manifest_path,
                verify_paths=not skip_manifest_path_verification,
            )
            val_summary_payload: dict | None = val_summary.to_dict()
        else:
            train_entries, val_entries = split_objaverse_entries(
                discovered_entries, args.val_ratio, args.seed
            )
            val_summary_payload = {
                "selected_entries": len(val_entries),
                "split_from_train": True,
                "val_ratio": args.val_ratio,
            }

        train_entries = _truncate_entries(train_entries, args.max_train_samples)
        val_entries = _truncate_entries(val_entries, args.max_val_samples)

        output_dir = Path(args.output_dir).expanduser().resolve()
        manifests_dir = output_dir / "manifests"
        if distributed.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            manifests_dir.mkdir(parents=True, exist_ok=True)
        _distributed_barrier(distributed)

        resolved_train_config_path = output_dir / "resolved_train_config.yaml"
        if distributed.is_main_process:
            resolved_train_config_path = _write_training_config_snapshot(output_dir, args)
        _distributed_barrier(distributed)

        if args.cache_dir is not None:
            cache_root = Path(args.cache_dir).expanduser().resolve()
            train_cache_dir = str(cache_root / "train")
            val_cache_dir = str(cache_root / "val")
        else:
            train_cache_dir = None
            val_cache_dir = None

        train_manifest_snapshot = manifests_dir / "train_entries.jsonl"
        val_manifest_snapshot = manifests_dir / "val_entries.jsonl"
        manifest_snapshot_status = "skipped" if args.skip_manifest_snapshot else "written"
        if distributed.is_main_process:
            reuse_manifest_snapshots = False
            if args.skip_manifest_snapshot:
                logging.info(
                    "Skipping run-directory train/val manifest snapshots by request."
                )
            else:
                reuse_manifest_snapshots = (
                    resume_fast_load_manifest
                    and train_manifest_snapshot.exists()
                    and (val_entries == [] or val_manifest_snapshot.exists())
                )
            if reuse_manifest_snapshots:
                manifest_snapshot_status = "reused"
                logging.info(
                    "Resume fast path enabled: reusing persisted manifest snapshots instead "
                    "of rewriting them into the run directory."
                )
            elif not args.skip_manifest_snapshot:
                _write_entries_manifest(train_manifest_snapshot, train_entries)
                _write_entries_manifest(val_manifest_snapshot, val_entries)
        _distributed_barrier(distributed)

        rank_suffix = f"_rank{distributed.rank:02d}" if distributed.enabled else ""
        train_dataset = ObjaverseDataset(
            root=args.data_root,
            manifest_path=args.manifest_path,
            point_samples=args.point_samples,
            cache_dir=train_cache_dir,
            entries=train_entries,
            bad_samples_path=str(output_dir / "logs" / f"bad_train_samples{rank_suffix}.jsonl"),
        )
        val_dataset = (
            ObjaverseDataset(
                root=args.data_root,
                manifest_path=args.val_manifest_path,
                point_samples=args.point_samples,
                cache_dir=val_cache_dir,
                entries=val_entries,
                bad_samples_path=str(output_dir / "logs" / f"bad_val_samples{rank_suffix}.jsonl"),
            )
            if val_entries
            else None
        )

        resume_batch_idx = int(resume_state.get("resume_batch_idx", 0)) if resume_state else 0
        resume_epoch = int(resume_state.get("epoch", 1)) if resume_state else 1
        train_sampler = None
        if distributed.enabled:
            train_sampler_cls = (
                ResumeDistributedSampler if resume_batch_idx > 0 else DistributedSampler
            )
            train_sampler = train_sampler_cls(
                train_dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
                **(
                    {
                        "resume_epoch": resume_epoch,
                        "resume_batch_idx": resume_batch_idx,
                        "batch_size": args.batch_size,
                    }
                    if train_sampler_cls is ResumeDistributedSampler
                    else {}
                ),
            )
        val_sampler = (
            DistributedSampler(
                val_dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=False,
                seed=args.seed + 1,
                drop_last=False,
            )
            if distributed.enabled and val_dataset is not None
            else None
        )

        train_generator = torch.Generator()
        train_generator.manual_seed(args.seed + distributed.rank)
        dataloader_multiprocessing_context = _resolve_dataloader_multiprocessing_context(
            device=device,
            num_workers=args.num_workers,
        )
        if dataloader_multiprocessing_context is not None and distributed.is_main_process:
            logging.info(
                "Using DataLoader multiprocessing_context=%s because trainer initialization "
                "loads CUDA state before worker startup.",
                dataloader_multiprocessing_context,
            )
        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_objaverse_batch,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            generator=train_generator,
            worker_init_fn=_seed_worker,
            multiprocessing_context=dataloader_multiprocessing_context,
        )

        val_dataloader = None
        if val_dataset is not None:
            val_generator = torch.Generator()
            val_generator.manual_seed(args.seed + 1 + distributed.rank)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=args.num_workers,
                collate_fn=collate_objaverse_batch,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
                generator=val_generator,
                worker_init_fn=_seed_worker,
                multiprocessing_context=dataloader_multiprocessing_context,
            )

        if distributed.is_main_process:
            logging.info("Train discovery summary: %s", discovered_summary.to_dict())
            logging.info("Train entries selected: %d", len(train_entries))
            if val_dataset is not None:
                logging.info("Validation summary: %s", val_summary_payload)
                logging.info("Validation entries selected: %d", len(val_entries))

        sample_eval_specs: list[SampleEvalSpec] = []
        sample_eval_specs_path = _sample_eval_specs_path(output_dir)
        if args.sample_eval_interval > 0 and args.sample_eval_max_samples > 0:
            sample_eval_entries = (
                []
                if args.sample_eval_manifest_path is not None
                else (val_entries if val_entries else train_entries)
            )
            sample_eval_spec_count = (
                len(sample_eval_entries)
                if args.sample_eval_resample_each_interval
                else args.sample_eval_max_samples
            )
            if distributed.is_main_process:
                sample_eval_specs = _resolve_sample_eval_specs(
                    output_dir=output_dir,
                    entries=sample_eval_entries,
                    max_samples=sample_eval_spec_count,
                    benchmark_manifest_path=args.sample_eval_manifest_path,
                    selection_seed=args.sample_eval_selection_seed,
                    resume_state_present=resume_state is not None,
                    bad_samples_path=output_dir
                    / "logs"
                    / "bad_sample_eval_specs.jsonl",
                )
                logging.info("Sample eval specs prepared: %d", len(sample_eval_specs))
            _distributed_barrier(distributed)
            sample_eval_specs = _load_sample_eval_specs(sample_eval_specs_path)
        else:
            if distributed.is_main_process:
                _save_sample_eval_specs(sample_eval_specs_path, [])
            _distributed_barrier(distributed)

        geometry_eval_specs: list[SampleEvalSpec] = []
        geometry_eval_fixed_specs: list[SampleEvalSpec] = []
        geometry_eval_specs_path = _geometry_eval_specs_path(output_dir)
        geometry_eval_fixed_specs_path = _geometry_eval_fixed_specs_path(output_dir)
        has_geometry_eval_source = bool(
            val_entries or args.geometry_eval_fixed_manifest_path is not None
        )
        if args.geometry_eval_interval > 0 and has_geometry_eval_source:
            geometry_eval_spec_count = (
                len(val_entries)
                if args.geometry_eval_resample_each_interval
                else args.geometry_eval_max_samples
            )
            if distributed.is_main_process:
                fixed_selection_seed = (
                    args.seed
                    if args.geometry_eval_fixed_selection_seed is None
                    else args.geometry_eval_fixed_selection_seed
                )
                geometry_eval_fixed_specs = _resolve_geometry_eval_fixed_specs(
                    output_dir=output_dir,
                    entries=val_entries,
                    max_samples=args.geometry_eval_fixed_max_samples,
                    benchmark_manifest_path=args.geometry_eval_fixed_manifest_path,
                    selection_seed=fixed_selection_seed,
                    resume_state_present=resume_state is not None,
                    bad_samples_path=output_dir
                    / "logs"
                    / "bad_geometry_eval_fixed_specs.jsonl",
                )
                if val_entries:
                    geometry_eval_specs = _resolve_geometry_eval_specs(
                        output_dir=output_dir,
                        entries=val_entries,
                        max_samples=geometry_eval_spec_count,
                        selection_seed=args.geometry_eval_selection_seed,
                        resume_state_present=resume_state is not None,
                        bad_samples_path=output_dir
                        / "logs"
                        / "bad_geometry_eval_specs.jsonl",
                    )
                else:
                    _save_sample_eval_specs(geometry_eval_specs_path, [])
                logging.info(
                    "Fixed geometry eval specs prepared: %d",
                    len(geometry_eval_fixed_specs),
                )
                logging.info("Geometry eval specs prepared: %d", len(geometry_eval_specs))
            _distributed_barrier(distributed)
            geometry_eval_fixed_specs = _load_sample_eval_specs(geometry_eval_fixed_specs_path)
            geometry_eval_specs = _load_sample_eval_specs(geometry_eval_specs_path)
        else:
            if distributed.is_main_process:
                if (
                    args.geometry_eval_interval > 0
                    and not val_entries
                    and args.geometry_eval_fixed_manifest_path is None
                ):
                    logging.warning(
                        "Geometry eval requested but no validation entries are available; disabling."
                    )
                _save_sample_eval_specs(geometry_eval_fixed_specs_path, [])
                _save_sample_eval_specs(geometry_eval_specs_path, [])
            _distributed_barrier(distributed)

        environment_metadata = _collect_environment_metadata(device, Path.cwd())
        environment_metadata["distributed"] = {
            "enabled": distributed.enabled,
            "backend": distributed.backend,
            "rank": distributed.rank,
            "local_rank": distributed.local_rank,
            "world_size": distributed.world_size,
        }
        budget_metadata = _training_budget_summary(args)

        if distributed.is_main_process:
            (output_dir / "dataset_summary.json").write_text(
                json.dumps(
                    {
                        "train_discovery_summary": discovered_summary.to_dict(),
                        "train_entries_selected": len(train_entries),
                        "train_manifest_snapshot": (
                            None
                            if args.skip_manifest_snapshot
                            else str(train_manifest_snapshot)
                        ),
                        "val_summary": val_summary_payload if val_dataset is not None else None,
                        "val_entries_selected": len(val_entries),
                        "val_manifest_snapshot": (
                            None
                            if args.skip_manifest_snapshot
                            else str(val_manifest_snapshot)
                        ),
                        "manifest_snapshot_status": manifest_snapshot_status,
                        "resolved_train_config_path": str(resolved_train_config_path),
                        "training_budget": budget_metadata,
                        "sample_eval_specs_path": str(sample_eval_specs_path),
                        "sample_eval_specs": [spec.to_dict() for spec in sample_eval_specs],
                        "sample_eval_manifest_path": args.sample_eval_manifest_path,
                        "geometry_eval_fixed_specs_path": str(geometry_eval_fixed_specs_path),
                        "geometry_eval_fixed_specs_count": len(geometry_eval_fixed_specs),
                        "geometry_eval_specs_path": str(geometry_eval_specs_path),
                        "geometry_eval_specs_count": len(geometry_eval_specs),
                    },
                    indent=2,
                    default=str,
                )
            )
        _distributed_barrier(distributed)

        final_checkpoint = trainer.fit(
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_steps=args.max_steps,
            log_interval=args.log_interval,
            val_interval=args.val_interval,
            val_max_batches=args.val_max_batches,
            sample_eval_specs=sample_eval_specs,
            sample_eval_interval=args.sample_eval_interval,
            sample_eval_max_samples=args.sample_eval_max_samples,
            sample_eval_selection_seed=args.sample_eval_selection_seed,
            sample_eval_resample_each_interval=args.sample_eval_resample_each_interval,
            geometry_eval_fixed_specs=geometry_eval_fixed_specs,
            geometry_eval_specs=geometry_eval_specs,
            geometry_eval_interval=args.geometry_eval_interval,
            geometry_eval_fixed_max_samples=args.geometry_eval_fixed_max_samples,
            geometry_eval_max_samples=args.geometry_eval_max_samples,
            geometry_eval_selection_seed=args.geometry_eval_selection_seed,
            geometry_eval_resample_each_interval=args.geometry_eval_resample_each_interval,
            geometry_eval_surface_samples=args.geometry_eval_surface_samples,
            geometry_eval_save_generated_meshes=args.geometry_eval_save_generated_meshes,
            sample_eval_resolution_base=args.sample_eval_resolution_base,
            sample_eval_chunk_size=args.sample_eval_chunk_size,
            sample_eval_top_p=args.sample_eval_top_p,
            sample_eval_guidance_scale=args.sample_eval_guidance_scale,
            sample_eval_num_diffusion_steps=args.sample_eval_num_diffusion_steps,
            sample_eval_sampling_strategy=args.sample_eval_sampling_strategy,
            sample_eval_render_multiview=args.sample_eval_render_multiview,
            sample_eval_render_nviews=args.sample_eval_render_nviews,
            sample_eval_render_resolution=args.sample_eval_render_resolution,
            sample_eval_render_backend=args.sample_eval_render_backend,
            save_model_only_interval=args.save_model_only_interval,
            save_full_state_interval=args.save_full_state_interval,
            save_final_trainer_state=args.save_final_trainer_state,
            resume_trainer_state=args.resume_trainer_state,
            tensorboard_dir=args.tensorboard_dir,
            enable_tensorboard=not args.disable_tensorboard,
            tensorboard_flush_secs=args.tensorboard_flush_secs,
            run_metadata={
                "cli_args": vars(args),
                "resume_state_present": resume_state is not None,
                "environment": environment_metadata,
                "training_budget": budget_metadata,
                "train_discovery_summary": discovered_summary.to_dict(),
                "train_entries_selected": len(train_entries),
                "train_manifest_snapshot": (
                    None if args.skip_manifest_snapshot else str(train_manifest_snapshot)
                ),
                "val_summary": val_summary_payload if val_dataset is not None else None,
                "val_entries_selected": len(val_entries),
                "val_manifest_snapshot": (
                    None if args.skip_manifest_snapshot else str(val_manifest_snapshot)
                ),
                "manifest_snapshot_status": manifest_snapshot_status,
                "resolved_train_config_path": str(resolved_train_config_path),
                "sample_eval_specs_path": str(sample_eval_specs_path),
                "sample_eval_specs": [spec.to_dict() for spec in sample_eval_specs],
            },
        )
        if distributed.is_main_process:
            logging.info("Training finished. Final checkpoint: %s", final_checkpoint)
    finally:
        _cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
