from block3d.training.data import (
    DatasetDiscoverySummary,
    ObjaverseDataset,
    SampleEvalSpec,
    collate_objaverse_batch,
    discover_objaverse_entries,
    prepare_sample_eval_specs,
    split_objaverse_entries,
)

__all__ = [
    "Block3DTrainer",
    "DatasetDiscoverySummary",
    "ObjaverseDataset",
    "SampleEvalSpec",
    "collate_objaverse_batch",
    "discover_objaverse_entries",
    "prepare_sample_eval_specs",
    "split_objaverse_entries",
]


def __getattr__(name: str):
    if name == "Block3DTrainer":
        from block3d.training.block_diffusion import Block3DTrainer

        return Block3DTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
