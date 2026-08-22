from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_CONFIG = REPO_ROOT / "block3d" / "configs" / "block3d.yaml"
BASE_TRAIN_CONFIG = REPO_ROOT / "block3d" / "configs" / "train_block3d.yaml"

ABLATIONS = {
    "b32": {"gpt_model.block_size": 32},
    "b96": {"gpt_model.block_size": 96},
    "b128": {"gpt_model.block_size": 128},
    "b256": {"gpt_model.block_size": 256},
    "t8": {"block_diffusion.num_diffusion_steps": 8},
    "t12": {"block_diffusion.num_diffusion_steps": 12},
    "t20": {"block_diffusion.num_diffusion_steps": 20},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the independently trained block-size and step ablations."
    )
    parser.add_argument(
        "--output-dir",
        default="block3d/configs/ablations",
        help="Directory for the generated model and training YAML files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, overrides in ABLATIONS.items():
        model_config = OmegaConf.load(BASE_MODEL_CONFIG)
        train_config = OmegaConf.load(BASE_TRAIN_CONFIG)
        for key, value in overrides.items():
            OmegaConf.update(model_config, key, value, merge=False)

        model_path = output_dir / f"block3d_{name}.yaml"
        train_path = output_dir / f"train_block3d_{name}.yaml"
        train_config.config_path = str(model_path.relative_to(REPO_ROOT)).replace("\\", "/")
        train_config.output_dir = f"runs/block3d_{name}"
        if name.startswith("t"):
            train_config.sample_eval_num_diffusion_steps = int(name[1:])

        OmegaConf.save(model_config, model_path)
        OmegaConf.save(train_config, train_path)
        print(f"wrote {model_path.relative_to(REPO_ROOT)}")
        print(f"wrote {train_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
