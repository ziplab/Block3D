import logging
from typing import Any, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

BOUNDING_BOX_MAX_SIZE = 1.925


def normalize_bbox(bounding_box_xyz: Tuple[float]):
    max_l = max(bounding_box_xyz)
    return [BOUNDING_BOX_MAX_SIZE * elem / max_l for elem in bounding_box_xyz]


def load_config(cfg_path: str) -> Any:
    """
    Load and resolve a configuration file.
    Args:
        cfg_path (str): The path to the configuration file.
    Returns:
        Any: The loaded and resolved configuration object.
    Raises:
        AssertionError: If the loaded configuration is not an instance of DictConfig.
    """

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg


def parse_structured(cfg_type: Any, cfg: DictConfig) -> Any:
    """
    Parses a configuration dictionary into a structured configuration object.
    Args:
        cfg_type (Any): The type of the structured configuration object.
        cfg (DictConfig): The configuration dictionary to be parsed.
    Returns:
        Any: The structured configuration object created from the dictionary.
    """

    scfg = OmegaConf.structured(cfg_type(**cfg))
    return scfg


def _materialize_shared_weight_aliases(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Backfill duplicated state-dict aliases used by shared submodules.

    Some public upstream checkpoints only store the top-level shared Fourier embedder
    as ``embedder.weight``. This repository's current module graph also exposes the
    same parameter under ``encoder.embedder.weight`` and
    ``occupancy_decoder.embedder.weight``. PyTorch expects all three keys when
    loading strictly, so we materialize the alias entries before calling
    ``load_state_dict``.
    """

    alias_map = {
        "embedder.weight": (
            "encoder.embedder.weight",
            "occupancy_decoder.embedder.weight",
        ),
    }
    model_keys = set(model.state_dict().keys())
    aliased: dict[str, torch.Tensor] = {}
    for source_key, alias_keys in alias_map.items():
        source_tensor = state_dict.get(source_key)
        if source_tensor is None:
            continue
        for alias_key in alias_keys:
            if alias_key in model_keys and alias_key not in state_dict:
                aliased[alias_key] = source_tensor
    if not aliased:
        return state_dict
    merged = dict(state_dict)
    merged.update(aliased)
    logging.info("Materialized shared checkpoint aliases: %s", sorted(aliased))
    return merged


def load_model_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    strict: bool = True,
    allowed_missing_keys: tuple[str, ...] = (),
) -> None:
    """
    Load a safetensors checkpoint into a PyTorch model.
    The model is updated in place.

    Args:
        model: PyTorch model to load weights into
        ckpt_path: Path to the safetensors checkpoint file

    Returns:
        None
    """
    assert ckpt_path.endswith(
        ".safetensors"
    ), f"Checkpoint path '{ckpt_path}' is not a safetensors file"

    state_dict = load_file(ckpt_path)
    state_dict = _materialize_shared_weight_aliases(model, state_dict)
    incompat = model.load_state_dict(state_dict, strict=strict)
    unexpected_keys = tuple(incompat.unexpected_keys)
    if unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys when loading {ckpt_path}: {list(unexpected_keys)}"
        )

    if strict:
        return

    missing_keys = tuple(incompat.missing_keys)
    disallowed_missing = sorted(set(missing_keys) - set(allowed_missing_keys))
    if disallowed_missing:
        raise RuntimeError(
            f"Missing keys when loading {ckpt_path}: {disallowed_missing}"
        )
    if missing_keys:
        logging.warning(
            "Ignoring expected missing keys when loading %s: %s",
            ckpt_path,
            list(missing_keys),
        )


def select_device() -> Any:
    """
    Selects the appropriate PyTorch device for tensor allocation.

    Returns:
        Any: The `torch.device` object.
    """
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
