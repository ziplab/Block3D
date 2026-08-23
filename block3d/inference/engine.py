from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

from block3d.inference.logits_postprocesses import sample_from_logits
from block3d.inference.utils import load_config, load_model_weights, parse_structured
from block3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder
from block3d.model.gpt.block_diffusion_utils import (
    BlockDiffusionTraceAccumulator,
    build_m2t_update_mask,
    build_t2t_update_mask,
    build_transfer_schedule,
    build_inference_shape_attention_mask,
    wrap_shape_attention_with_condition_prefix,
)
from block3d.model.gpt.dual_stream_roformer import DualStreamRoformer


class Engine:
    def __init__(
        self,
        config_path: str,
        gpt_ckpt_path: str,
        shape_ckpt_path: str,
        device: torch.device,
    ):
        """
        Initializes the inference engine with the given configuration and checkpoint paths.
        Args:
            config_path (str): Path to the configuration file.
            gpt_ckpt_path (str): Path to the GPT model checkpoint file.
            shape_ckpt_path (str): Path to the shape model checkpoint file.
            device (torch.device): The device to run the models on (e.g., 'cpu' or 'cuda').
        Attributes:
            cfg (dict): Loaded configuration from the config file.
            device (torch.device): The device to run the models on.
            gpt_model (DualStreamRoformer): The GPT model initialized and loaded with weights.
            shape_model (OneDAutoEncoder): The shape model initialized and loaded with weights.
            text_model (CLIPTextModelWithProjection): The text model initialized from a pretrained model.
            text_tokenizer: The tokenizer for the text model.
            max_new_tokens (int): Maximum number of new tokens for the shape model.
            min_id (int): Minimum ID for the shape model codes.
            max_id (int): Maximum ID for the shape model codes.
        """

        self.cfg = load_config(config_path)
        self.device = device

        self.gpt_model = DualStreamRoformer(
            parse_structured(DualStreamRoformer.Config, self.cfg.gpt_model)
        )
        load_model_weights(
            self.gpt_model,
            gpt_ckpt_path,
        )
        self.gpt_model = self.gpt_model.eval().to(self.device)

        self.shape_model = OneDAutoEncoder(
            parse_structured(OneDAutoEncoder.Config, self.cfg.shape_model)
        )
        load_model_weights(
            self.shape_model,
            shape_ckpt_path,
            strict=False,
            allowed_missing_keys=(
                "encoder.embedder.weight",
                "occupancy_decoder.embedder.weight",
            ),
        )
        self.shape_model = self.shape_model.eval().to(self.device)

        # copy vq codebook to gpt
        with torch.no_grad():
            codebook = self.shape_model.bottleneck.block.get_codebook()
            codebook = self.gpt_model.shape_proj(codebook).detach()
        self.gpt_model.transformer.wte.weight.data[: codebook.shape[0]] = codebook

        text_model_path = Path(
            str(self.cfg.text_model_pretrained_model_name_or_path)
        ).expanduser().resolve()
        if not text_model_path.is_dir():
            raise FileNotFoundError(f"CLIP model directory not found: {text_model_path}")
        self.text_model = CLIPTextModelWithProjection.from_pretrained(
            str(text_model_path),
            force_download=False,
            local_files_only=True,
        ).to(self.device).eval()
        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(
            str(text_model_path),
            local_files_only=True,
        )

        self.max_new_tokens = self.shape_model.cfg.num_encoder_latents
        self.min_id = 0
        self.max_id = self.shape_model.cfg.num_codes
        self.generation_mode = getattr(self.gpt_model.cfg, "generation_mode", "block_diffusion")
        if self.generation_mode != "block_diffusion":
            raise ValueError(
                "This engine only supports generation_mode='block_diffusion'."
            )
        self.block_size = getattr(self.gpt_model.cfg, "block_size", 1)
        self.mask_token_id = self.gpt_model.shape_mask_id
        block_diffusion_cfg = self.cfg.get("block_diffusion")
        self.default_sampling_strategy = (
            str(block_diffusion_cfg.sampling_strategy)
            if block_diffusion_cfg is not None and "sampling_strategy" in block_diffusion_cfg
            else "block3d"
        )
        self.m2t_threshold = (
            float(block_diffusion_cfg.m2t_threshold)
            if block_diffusion_cfg is not None and "m2t_threshold" in block_diffusion_cfg
            else 0.95
        )
        self.t2t_threshold = (
            float(block_diffusion_cfg.t2t_threshold)
            if block_diffusion_cfg is not None
            and "t2t_threshold" in block_diffusion_cfg
            else 0.9
        )
        self.enable_t2t = bool(
            block_diffusion_cfg.enable_t2t
            if block_diffusion_cfg is not None and "enable_t2t" in block_diffusion_cfg
            else True
        )
        self.use_block_diffusion_prefix_cache = bool(
            getattr(self.gpt_model.cfg, "use_block_diffusion_prefix_cache", True)
        )
        self.last_block_diffusion_sampling_summary: Optional[dict[str, object]] = None

    @torch.inference_mode()
    def prepare_conditions_with_bbox(
        self,
        cond: torch.Tensor,
        bounding_box_tensor: Optional[torch.Tensor] = None,
    ):
        """
        Prepares condition embeddings by incorporating bounding box information.

        Concatenates a bounding-box embedding only when a box is explicitly provided and
        the model supports bounding-box projection.

        Args:
            cond (torch.Tensor): The input condition embeddings tensor of shape (B, seq_len, dim).
            bounding_box_xyz (Optional[torch.Tensor], optional): The size of the bounding box
                as (x, y, z) dimensions represented as a tensor. If None, text-only
                conditioning is used and no extra condition token is appended.

        Returns:
            torch.Tensor: The condition tensor with bounding box embeddings concatenated along
                the sequence dimension if bounding box projection is supported, otherwise
                returns the original condition tensor unchanged.
        """
        if not hasattr(self.gpt_model, "bbox_proj") or bounding_box_tensor is None:
            return cond

        bounding_box_tensor = bounding_box_tensor.to(device=self.device, dtype=cond.dtype)
        bbox_emb = self.gpt_model.bbox_proj(bounding_box_tensor).unsqueeze(dim=1)
        cond = torch.cat([cond, bbox_emb], dim=1)
        return cond

    @torch.inference_mode()
    def prepare_conditions(
        self,
        prompts: list[str],
        bounding_box_xyz: Optional[Tuple[float]] = None,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        prompt_embeds = self.run_clip(prompts)
        if bounding_box_xyz is not None:
            cond_bbox = torch.atleast_2d(torch.tensor(bounding_box_xyz)).to(self.device)
            uncond_bbox = torch.zeros_like(cond_bbox).to(self.device)
        else:
            cond_bbox = None
            uncond_bbox = None

        cond = self.prepare_conditions_with_bbox(prompt_embeds, cond_bbox)
        if guidance_scale > 0.0:
            uncond_embeds = self.run_clip([""] * len(prompts))
            uncond = self.prepare_conditions_with_bbox(uncond_embeds, uncond_bbox)
            cond = torch.cat([cond, uncond], dim=0)
        return cond

    @torch.inference_mode()
    def run_clip(self, text_inputs):
        """
        Processes the given text inputs using a text tokenizer and a text model, and returns the encoded text embeddings.
        Args:
            text_inputs (str or List[str]): The input text or list of texts to be processed.
        Returns:
            torch.Tensor: The encoded text embeddings.
        """

        text_inputs = self.text_tokenizer(
            text_inputs,
            max_length=self.text_tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            # use full precision for text encoder
            with torch.autocast(device_type=self.device.type, enabled=False):
                encoded = self.text_model(**text_inputs)
            if self.gpt_model.cfg.use_pooled_text_embed:
                embed = encoded.text_embeds.unsqueeze(1)  # [bs, 1, 512]
            else:
                embed = encoded.last_hidden_state  # [bs, 77, 512]
        embed = self.gpt_model.encode_text(embed)

        return embed

    @torch.inference_mode()
    def encode_shape_tokens(self, shape_ids: torch.Tensor) -> torch.Tensor:
        return self.gpt_model.encode_token(shape_ids)

    def _duplicate_cfg_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor, tensor], dim=0)

    def _block_diffusion_cfg_gamma(
        self,
        guidance_scale: float,
        step_idx: int,
        total_steps: int,
    ) -> float:
        total_steps = max(int(total_steps), 1)
        clamped_step_idx = min(max(int(step_idx), 0), total_steps - 1)
        return float(guidance_scale) * float(total_steps - clamped_step_idx) / float(
            total_steps
        )

    def _apply_block_diffusion_cfg(
        self,
        block_logits: torch.Tensor,
        guidance_scale: float,
        step_idx: int,
        total_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if guidance_scale <= 0.0:
            return block_logits, block_logits

        cond_logits, uncond_logits = block_logits.float().chunk(2, dim=0)
        gamma = self._block_diffusion_cfg_gamma(
            guidance_scale=guidance_scale,
            step_idx=step_idx,
            total_steps=total_steps,
        )
        guided_logits = (1.0 + gamma) * cond_logits - gamma * uncond_logits
        return guided_logits, cond_logits

    def _resolve_sampling_strategy(self, sampling_strategy: Optional[str]) -> str:
        strategy = (
            self.default_sampling_strategy if sampling_strategy is None else str(sampling_strategy)
        )
        if strategy != "block3d":
            raise ValueError(
                f"Unsupported sampling_strategy={strategy!r}; expected 'block3d'"
            )
        return strategy


    @torch.inference_mode()
    def run_block_diffusion_gpt(
        self,
        prompts: list[str],
        num_steps: int = 4,
        top_p: float = None,
        bounding_box_xyz: Optional[Tuple[float]] = None,
        return_denoise_trace: bool = False,
        sampling_strategy: Optional[str] = None,
        guidance_scale: float = 3.0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        cond = self.prepare_conditions(
            prompts,
            bounding_box_xyz,
            guidance_scale=guidance_scale,
        )
        return self.run_block_diffusion_gpt_from_prepared(
            cond=cond,
            num_steps=num_steps,
            top_p=top_p,
            return_denoise_trace=return_denoise_trace,
            sampling_strategy=sampling_strategy,
            guidance_scale=guidance_scale,
        )

    @torch.inference_mode()
    def run_block_diffusion_gpt_from_prepared(
        self,
        cond: torch.Tensor,
        num_steps: int = 4,
        top_p: float = None,
        return_denoise_trace: bool = False,
        sampling_strategy: Optional[str] = None,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        uses_guidance = guidance_scale > 0.0
        model_batch_size = cond.shape[0]
        if uses_guidance:
            if model_batch_size % 2 != 0:
                raise ValueError(
                    "Block-diffusion CFG expects prepared cond to contain concatenated "
                    f"conditional/unconditional branches, got batch size {model_batch_size}"
                )
            batch_size = model_batch_size // 2
        else:
            batch_size = model_batch_size
        resolved_sampling_strategy = self._resolve_sampling_strategy(sampling_strategy)
        shape_ids = torch.full(
            (batch_size, self.max_new_tokens),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        forward_call_count = 0
        prefix_cache_rebuild_count = 0
        trace_accumulator = (
            BlockDiffusionTraceAccumulator(
                batch_size=batch_size,
                num_shape_tokens=self.max_new_tokens,
                block_size=self.block_size,
                requested_num_diffusion_steps=num_steps,
            )
            if return_denoise_trace
            else None
        )
        use_prefix_cache = self.use_block_diffusion_prefix_cache and (
            not getattr(self.gpt_model.cfg, "use_single_blocks_in_diffusion", False)
        )

        for block_start in range(0, self.max_new_tokens, self.block_size):
            block_end = min(block_start + self.block_size, self.max_new_tokens)
            curr_block_size = block_end - block_start
            shape_mask = build_inference_shape_attention_mask(
                context_len=block_start,
                block_len=curr_block_size,
                device=self.device,
            )
            attn_mask = wrap_shape_attention_with_condition_prefix(
                shape_mask,
                cond.shape[1],
            )
            full_shape_position_ids = torch.arange(
                block_start + curr_block_size,
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(model_batch_size, -1)

            prefix_cache = None
            if use_prefix_cache:
                prefix_embed = self.encode_shape_tokens(shape_ids[:, :block_start])
                if uses_guidance:
                    prefix_embed = self._duplicate_cfg_batch(prefix_embed)
                prefix_cache = self.gpt_model.build_block_diffusion_prefix_cache(
                    prefix_embed=prefix_embed,
                    cond=cond,
                )
                prefix_cache_rebuild_count += 1

            def compute_block_logits(
                block_ids: torch.Tensor,
                guidance_step_idx: int,
                guidance_total_steps: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if prefix_cache is None:
                    shape_input_ids = torch.cat(
                        [shape_ids[:, :block_start], block_ids], dim=1
                    )
                    if uses_guidance:
                        shape_input_ids = self._duplicate_cfg_batch(shape_input_ids)
                    shape_embed = self.encode_shape_tokens(shape_input_ids)
                    logits = self.gpt_model.forward_block_diffusion(
                        embed=shape_embed,
                        cond=cond,
                        attn_mask=attn_mask,
                        shape_position_ids=full_shape_position_ids,
                    )
                    block_logits = logits[:, -curr_block_size:, self.min_id : self.max_id]
                    return self._apply_block_diffusion_cfg(
                        block_logits=block_logits,
                        guidance_scale=guidance_scale,
                        step_idx=guidance_step_idx,
                        total_steps=guidance_total_steps,
                    )

                if uses_guidance:
                    block_ids = self._duplicate_cfg_batch(block_ids)
                block_embed = self.encode_shape_tokens(block_ids)
                logits = self.gpt_model.forward_block_diffusion_with_prefix_cache(
                    embed=block_embed,
                    prefix_cache=prefix_cache,
                    attn_mask=attn_mask,
                    shape_position_ids=full_shape_position_ids,
                )
                block_logits = logits[..., self.min_id : self.max_id]
                return self._apply_block_diffusion_cfg(
                    block_logits=block_logits,
                    guidance_scale=guidance_scale,
                    step_idx=guidance_step_idx,
                    total_steps=guidance_total_steps,
                )

            if resolved_sampling_strategy == "block3d":
                block3d_num_steps = max(int(num_steps), 1)
                transfer_schedule = build_transfer_schedule(
                    block_length=curr_block_size,
                    num_steps=block3d_num_steps,
                )
                local_step_idx = 0
                for step_idx in range(block3d_num_steps):
                    block_ids = shape_ids[:, block_start:block_end]
                    forward_call_count += 1
                    if trace_accumulator is not None:
                        trace_accumulator.note_forward_call()
                    guided_logits, conditional_logits = compute_block_logits(
                        block_ids,
                        guidance_step_idx=step_idx,
                        guidance_total_steps=block3d_num_steps,
                    )
                    candidate_ids, _ = sample_from_logits(
                        guided_logits,
                        top_p=top_p,
                    )
                    candidate_ids = candidate_ids.squeeze(-1)
                    candidate_probs = torch.softmax(
                        conditional_logits.float(), dim=-1
                    ).gather(
                        dim=-1,
                        index=candidate_ids.unsqueeze(-1),
                    )
                    candidate_probs = candidate_probs.squeeze(-1)

                    mask_positions = block_ids.eq(self.mask_token_id)
                    m2t_mask = build_m2t_update_mask(
                        mask_positions=mask_positions,
                        candidate_probs=candidate_probs,
                        required_transfer_tokens=int(transfer_schedule[step_idx].item()),
                        confidence_threshold=self.m2t_threshold,
                    )
                    t2t_mask = (
                        build_t2t_update_mask(
                            block_ids=block_ids,
                            candidate_ids=candidate_ids,
                            candidate_probs=candidate_probs,
                            mask_token_id=self.mask_token_id,
                            editing_threshold=self.t2t_threshold,
                        )
                        if self.enable_t2t
                        else torch.zeros_like(m2t_mask)
                    )
                    update_mask = m2t_mask | t2t_mask
                    if not bool(update_mask.any().item()) and not bool(mask_positions.any().item()):
                        break
                    block_ids = torch.where(update_mask, candidate_ids, block_ids)
                    shape_ids[:, block_start:block_end] = block_ids
                    if trace_accumulator is not None:
                        trace_accumulator.record_step(
                            block_start=block_start,
                            local_step_idx=local_step_idx,
                            block_ids=block_ids,
                            mask_token_id=self.mask_token_id,
                        )
                    local_step_idx += 1
                continue

        self.last_block_diffusion_sampling_summary = {
            "total_forward_calls": int(forward_call_count),
            "requested_num_diffusion_steps": int(num_steps),
            "batch_size": int(batch_size),
            "num_shape_tokens": int(self.max_new_tokens),
            "block_size": int(self.block_size),
            "sampling_strategy": resolved_sampling_strategy,
            "used_prefix_cache": bool(use_prefix_cache),
            "prefix_cache_rebuild_count": int(prefix_cache_rebuild_count),
            "guidance_scale": float(guidance_scale),
        }
        if trace_accumulator is None:
            return shape_ids
        return shape_ids, trace_accumulator.finalize()

    @torch.inference_mode()
    def run_shape_decode(
        self,
        output_ids: torch.Tensor,
        resolution_base: float = 8.0,
        chunk_size: int = 100_000,
        use_warp: bool = True,
    ):
        """
        Decodes the shape from the given output IDs and extracts the geometry.
        Args:
            output_ids (torch.Tensor): The tensor containing the output IDs.
            resolution_base (float, optional): The base resolution for geometry extraction. Defaults to 8.43.
            chunk_size (int, optional): The chunk size for processing. Defaults to 100,000.
        Returns:
            tuple: A tuple containing the vertices and faces of the mesh.
        """
        shape_ids = (
            output_ids[:, : self.shape_model.cfg.num_encoder_latents, ...]
            .clamp_(0, self.shape_model.cfg.num_codes - 1)
            .view(-1, self.shape_model.cfg.num_encoder_latents)
        )
        latents = self.shape_model.decode_indices(shape_ids)
        mesh_v_f, _ = self.shape_model.extract_geometry(
            latents,
            resolution_base=resolution_base,
            chunk_size=chunk_size,
            use_warp=use_warp,
        )
        return mesh_v_f

    @torch.inference_mode()
    def t2s(
        self,
        prompts: list[str],
        guidance_scale: float = 3.0,
        resolution_base: float = 8.0,
        chunk_size: int = 100_000,
        top_p: float = None,
        bounding_box_xyz: Optional[Tuple[float]] = None,
        num_diffusion_steps: int = 4,
        sampling_strategy: Optional[str] = None,
    ):
        """
        Generates a 3D mesh from text prompts using a GPT model and shape decoder.
        Args:
            prompts (list[str]): A list of text prompts to guide the generation.
            guidance_scale (float, optional): The scale of guidance for the GPT model. Default is 3.0.
            resolution_base (float, optional): The base resolution for the shape decoder. Default is 8.0.
            chunk_size (int, optional): The chunk size for processing the shape decoding. Default is 100,000.
            top_p (float, optional): The cumulative probability threshold for nucleus sampling.
                If None, argmax selection is performed. Otherwise, the smallest token set with cumulative probability >= top_p is retained.
            bounding_box_xyz (Tuple[float] | None, optional): The size of the bounding box for the generated mesh
                as (x, y, z) dimensions. Each value must be between 0 and 1.925. If None,
                uses default bounding box sizing.
        Returns:
            mesh_v_f: The generated 3D mesh vertices and faces.
        """
        output_ids = self.run_block_diffusion_gpt(
            prompts=prompts,
            num_steps=num_diffusion_steps,
            top_p=top_p,
            bounding_box_xyz=bounding_box_xyz,
            sampling_strategy=sampling_strategy,
            guidance_scale=guidance_scale,
        )
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            mesh_v_f = self.run_shape_decode(output_ids, resolution_base, chunk_size)
        return mesh_v_f
