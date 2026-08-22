from __future__ import annotations

import math
from typing import Optional

import torch


def _num_blocks(num_tokens: int, block_size: int) -> int:
    return math.ceil(num_tokens / block_size)


def _quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    if lower_idx == upper_idx:
        return sorted_values[lower_idx]
    frac = position - lower_idx
    lower = sorted_values[lower_idx]
    upper = sorted_values[upper_idx]
    return lower * (1.0 - frac) + upper * frac


class BlockDiffusionTraceAccumulator:
    """
    Collect per-block local denoising statistics without changing sampling behavior.

    Statistics are tracked in a block-local step coordinate system:
    step 1 means "after the first update inside each block", step 2 means the
    second update inside each block, and so on. This lets us answer questions
    like "90% of blocks finished by which denoising step?" across all blocks.
    """

    def __init__(
        self,
        batch_size: int,
        num_shape_tokens: int,
        block_size: int,
        requested_num_diffusion_steps: Optional[int],
    ) -> None:
        self.batch_size = int(batch_size)
        self.num_shape_tokens = int(num_shape_tokens)
        self.block_size = int(block_size)
        self.requested_num_diffusion_steps = (
            None
            if requested_num_diffusion_steps is None
            else int(requested_num_diffusion_steps)
        )
        self.total_blocks = _num_blocks(self.num_shape_tokens, self.block_size)
        self.total_forward_calls = 0
        self._per_block_finished_step = torch.full(
            (self.batch_size, self.total_blocks),
            fill_value=-1,
            dtype=torch.long,
        )
        self._per_step_remaining_mask_counts: list[torch.Tensor] = []

    def note_forward_call(self) -> None:
        self.total_forward_calls += 1

    def _ensure_step_capacity(self, size: int) -> None:
        while len(self._per_step_remaining_mask_counts) < size:
            self._per_step_remaining_mask_counts.append(
                torch.zeros(self.batch_size, dtype=torch.long)
            )

    def record_step(
        self,
        block_start: int,
        local_step_idx: int,
        block_ids: torch.Tensor,
        mask_token_id: int,
    ) -> None:
        if local_step_idx < 0:
            raise ValueError(f"local_step_idx must be non-negative, got {local_step_idx}")
        block_index = block_start // self.block_size
        if not 0 <= block_index < self.total_blocks:
            raise ValueError(
                f"block_start={block_start} resolved to invalid block index {block_index}"
            )

        self._ensure_step_capacity(local_step_idx + 1)
        remaining_mask_count = (
            block_ids.eq(mask_token_id).sum(dim=1).to(device="cpu", dtype=torch.long)
        )
        self._per_step_remaining_mask_counts[local_step_idx] += remaining_mask_count

        newly_finished = (
            remaining_mask_count.eq(0)
            & self._per_block_finished_step[:, block_index].lt(0)
        )
        if bool(newly_finished.any().item()):
            self._per_block_finished_step[newly_finished, block_index] = local_step_idx + 1

    def finalize(self) -> dict[str, object]:
        max_local_steps_observed = len(self._per_step_remaining_mask_counts)
        finished_steps = self._per_block_finished_step.clone()
        unresolved = finished_steps.lt(0)
        if bool(unresolved.any().item()):
            fill_value = max(max_local_steps_observed, 1)
            finished_steps[unresolved] = fill_value

        samples: list[dict[str, object]] = []
        for sample_idx in range(self.batch_size):
            sample_finished_steps = [int(v) for v in finished_steps[sample_idx].tolist()]
            per_step_remaining_mask_count = [
                int(step_counts[sample_idx].item())
                for step_counts in self._per_step_remaining_mask_counts
            ]
            per_step_remaining_mask_ratio = [
                float(count) / float(max(self.num_shape_tokens, 1))
                for count in per_step_remaining_mask_count
            ]
            per_step_finished_block_count = [
                sum(1 for value in sample_finished_steps if value <= step_idx + 1)
                for step_idx in range(max_local_steps_observed)
            ]
            per_step_finished_block_fraction = [
                float(count) / float(max(self.total_blocks, 1))
                for count in per_step_finished_block_count
            ]
            samples.append(
                {
                    "sample_idx": sample_idx,
                    "per_block_finished_step": sample_finished_steps,
                    "per_step_remaining_mask_count": per_step_remaining_mask_count,
                    "per_step_remaining_mask_ratio": per_step_remaining_mask_ratio,
                    "per_step_finished_block_count": per_step_finished_block_count,
                    "per_step_finished_block_fraction": per_step_finished_block_fraction,
                    "finished_step_mean": (
                        float(sum(sample_finished_steps)) / float(len(sample_finished_steps))
                        if sample_finished_steps
                        else 0.0
                    ),
                    "finished_step_p50": _quantile(sample_finished_steps, 0.50),
                    "finished_step_p75": _quantile(sample_finished_steps, 0.75),
                    "finished_step_p90": _quantile(sample_finished_steps, 0.90),
                }
            )

        return {
            "batch_size": self.batch_size,
            "num_shape_tokens": self.num_shape_tokens,
            "block_size": self.block_size,
            "total_blocks": self.total_blocks,
            "requested_num_diffusion_steps": self.requested_num_diffusion_steps,
            "max_local_steps_observed": max_local_steps_observed,
            "total_forward_calls": self.total_forward_calls,
            "samples": samples,
        }


def duplicate_shape_position_ids(
    batch_size: int,
    num_shape_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    position_ids = torch.arange(num_shape_tokens, device=device, dtype=torch.long)
    position_ids = torch.cat([position_ids, position_ids], dim=0)
    return position_ids.unsqueeze(0).expand(batch_size, -1)


def build_training_shape_attention_mask(
    num_shape_tokens: int,
    block_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build the shape-side attention mask for [clean_shape | noisy_shape].

    clean block k:
      - can attend to clean blocks <= k
      - cannot attend to noisy half

    noisy block k:
      - can attend to clean blocks < k
      - can attend bidirectionally within noisy block k
      - cannot attend to clean block k or future clean blocks
      - cannot attend to any other noisy block
    """
    total_shape_len = num_shape_tokens * 2
    mask = torch.zeros(
        (total_shape_len, total_shape_len), dtype=torch.bool, device=device
    )
    num_blocks = _num_blocks(num_shape_tokens, block_size)

    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, num_shape_tokens)

        clean_start = block_start
        clean_end = block_end
        noisy_start = num_shape_tokens + block_start
        noisy_end = num_shape_tokens + block_end

        # clean block k sees clean blocks <= k
        mask[clean_start:clean_end, :clean_end] = True

        # noisy block k sees clean blocks < k
        mask[noisy_start:noisy_end, :clean_start] = True
        # noisy block k sees itself bidirectionally
        mask[noisy_start:noisy_end, noisy_start:noisy_end] = True

    return mask


def build_inference_shape_attention_mask(
    context_len: int,
    block_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build the shape-side attention mask for [context_shape | x_t_block].

    - context is causal and fully visible to the current block
    - current block is bidirectional within the block
    - current block does not see future tokens because they are absent
    """
    total_shape_len = context_len + block_len
    mask = torch.zeros(
        (total_shape_len, total_shape_len), dtype=torch.bool, device=device
    )
    if context_len > 0:
        mask[:context_len, :context_len] = torch.tril(
            torch.ones(
                (context_len, context_len), dtype=torch.bool, device=device
            )
        )
        mask[context_len:, :context_len] = True
    mask[context_len:, context_len:] = True
    return mask


def wrap_shape_attention_with_condition_prefix(
    shape_mask: torch.Tensor,
    cond_len: int,
) -> torch.Tensor:
    """
    Expand a shape-side mask into a full [cond | shape] mask.

    cond rows only attend to cond columns.
    shape rows attend to all cond columns plus the shape-side mask.
    """
    total_len = cond_len + shape_mask.shape[0]
    full_mask = torch.zeros(
        (total_len, total_len), dtype=torch.bool, device=shape_mask.device
    )
    if cond_len > 0:
        full_mask[:cond_len, :cond_len] = True
        full_mask[cond_len:, :cond_len] = True
    full_mask[cond_len:, cond_len:] = shape_mask
    return full_mask


def sample_block_timesteps(
    batch_size: int,
    num_shape_tokens: int,
    block_size: int,
    t_min: float,
    t_max: float,
    device: torch.device,
) -> torch.Tensor:
    num_blocks = _num_blocks(num_shape_tokens, block_size)
    t = torch.rand((batch_size, num_blocks), device=device)
    t = t * (t_max - t_min) + t_min
    return t.repeat_interleave(block_size, dim=1)[:, :num_shape_tokens]


def sample_random_wrong_tokens(
    clean_shape_ids: torch.Tensor,
    num_codes: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample random replacement tokens that are guaranteed to differ from the clean ids.
    """
    if num_codes <= 1:
        raise ValueError(f"num_codes must be > 1 to sample wrong tokens, got {num_codes}")

    clipped_ids = clean_shape_ids.clamp(min=0, max=num_codes - 1)
    wrong_ids = torch.randint(
        low=0,
        high=num_codes - 1,
        size=clean_shape_ids.shape,
        device=clean_shape_ids.device,
        dtype=clean_shape_ids.dtype,
    )
    wrong_ids = wrong_ids + wrong_ids.ge(clipped_ids).to(dtype=wrong_ids.dtype)
    if valid_mask is not None:
        wrong_ids = torch.where(valid_mask, wrong_ids, clean_shape_ids)
    return wrong_ids


def build_transfer_schedule(
    block_length: int,
    num_steps: int,
) -> torch.Tensor:
    """
    Build the fixed per-step transfer budget used by Block3D.

    Each step is assigned a fixed minimum number of masked tokens to reveal so
    that a block can be fully resolved within the configured denoising budget.
    """
    if block_length < 0:
        raise ValueError(f"block_length must be non-negative, got {block_length}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if block_length == 0:
        return torch.zeros((num_steps,), dtype=torch.long)

    base = block_length // num_steps
    remainder = block_length % num_steps
    schedule = torch.full((num_steps,), base, dtype=torch.long)
    schedule[:remainder] += 1
    return schedule


def build_m2t_update_mask(
    mask_positions: torch.Tensor,
    candidate_probs: torch.Tensor,
    required_transfer_tokens: int,
    confidence_threshold: float,
) -> torch.Tensor:
    """
    Select masked positions using Block3D's confidence-guided M2T rule.

    Positions above the confidence threshold are accepted directly. If that does not
    satisfy the transfer budget of the current denoising step, the highest-confidence
    masked positions are additionally selected to meet that budget.
    """
    if required_transfer_tokens < 0:
        raise ValueError(
            "required_transfer_tokens must be non-negative, got "
            f"{required_transfer_tokens}"
        )

    update_mask = mask_positions & candidate_probs.gt(confidence_threshold)
    batch_size = mask_positions.shape[0]
    for batch_idx in range(batch_size):
        active = torch.nonzero(mask_positions[batch_idx], as_tuple=False).flatten()
        if active.numel() == 0:
            continue

        required = min(int(required_transfer_tokens), int(active.numel()))
        if required == 0:
            continue
        current_selected = int(update_mask[batch_idx, active].sum().item())
        if current_selected >= required:
            continue

        # ``active`` is in ascending global-position order. Stable sorting therefore
        # implements the paper's deterministic lower-position tie break.
        ranked = torch.argsort(
            candidate_probs[batch_idx, active],
            dim=0,
            descending=True,
            stable=True,
        )
        chosen = active[ranked[:required]]
        update_mask[batch_idx, chosen] = True
    return update_mask


def build_t2t_update_mask(
    block_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_probs: torch.Tensor,
    mask_token_id: int,
    editing_threshold: float,
) -> torch.Tensor:
    """
    Select editable non-mask positions for token-to-token correction.
    """
    return (
        block_ids.ne(mask_token_id)
        & candidate_ids.ne(block_ids)
        & candidate_probs.gt(editing_threshold)
    )
