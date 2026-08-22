# Block3D

Official implementation of **Block3D: Efficient Text-to-3D Generation via
Block-Wise Diffusion**.

[Project page](https://alexandertsui.github.io/block3d/) ·
[Code](https://github.com/AlexanderTsui/Block3D)

Block3D adapts a discrete 3D shape prior to block-causal diffusion. It generates
contiguous shape-token blocks, denoises each active block in parallel, and edits
low-confidence tokens before the block is committed.

## Installation

The reference environment is Linux, Python 3.10+, PyTorch 2.2+, CUDA, and four
A100 80GB GPUs for training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`pymeshlab` is optional. Blender is only needed for mesh formats that cannot be
loaded directly by `trimesh`. PyTorch3D is required for the eight-view CLIPScore
command.

## Checkpoints and Data

Weights and TRELLIS-500K data are not included. Restore the released Cube v0.5
initialization and the local CLIP model using this layout:

```text
model_weights/
  shape_gpt.safetensors
  shape_tokenizer.safetensors
  clip-vit-large-patch14/
data/trellis500k_train/
  source_manifest.jsonl
  pairs/
  objaverse_xl_github/
  objaverse_xl_sketchfab/
  abo/
```

Training records use `mesh_path`, `text`, and `pair_path`. Each pair file must
contain `shape_ids`, `bbox_xyz`, `text_input_ids`, and `text_attention_mask`.
The fixed 100-object evaluation definition is included as
`random100_evaluation_manifest.jsonl`.

The source package cannot reproduce the reported run without these external
weights and data. During fine-tuning, the shape tokenizer, codebook, CLIP text
encoder, and shape decoder remain frozen; only the text-to-shape generator is
updated.

## Prepare the 300K Split

```bash
python scripts/prepare_block_diffusion_eval_split.py \
  --dataset-root data/trellis500k_train \
  --source-manifest-path data/trellis500k_train/source_manifest.jsonl \
  --eval-manifest-path random100_evaluation_manifest.jsonl \
  --output-dir data/splits \
  --train-count 300000 \
  --val-ratio 0 \
  --seed 42
```

This excludes the fixed 100 evaluation records and writes
`data/splits/train_manifest.jsonl` for the training commands below.

## Training

Main Block3D model:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m block3d.train_block_diffusion \
  --train-config-path block3d/configs/train_block3d.yaml
```

M2T-only ablation:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m block3d.train_block_diffusion \
  --train-config-path block3d/configs/train_block3d_m2t_only.yaml
```

Both runs use 35K optimizer steps, global batch size 40, bfloat16, seed 42,
AdamW with learning rate `1e-4`, and text-only conditioning. The final generator
checkpoint is written to `runs/<run>/checkpoints/gpt_final.safetensors`.

The block-size and denoising-step ablation configurations can be materialized
with:

```bash
python scripts/materialize_ablation_configs.py
```

## Inference

```bash
python -m block3d.generate \
  --config-path block3d/configs/block3d.yaml \
  --gpt-ckpt-path runs/block3d/checkpoints/gpt_final.safetensors \
  --shape-ckpt-path model_weights/shape_tokenizer.safetensors \
  --prompt "a wooden chair" \
  --output-dir outputs/chair \
  --num-diffusion-steps 4 \
  --guidance-scale 3.0 \
  --sampling-strategy block3d \
  --disable-postprocessing
```

The output mesh is `outputs/chair/output.obj`. Do not pass
`--bounding-box-xyz` for the reported text-only protocol. Batch-generation
orchestration is intentionally not included.

## Evaluation

Create `outputs/eval100/samples.jsonl` with one record per generated mesh:

```json
{"sample_id":"0000","prompt_text":"...","reference_mesh_path":"...","generated_mesh_path":"0000/generated.obj","bbox_xyz":[0.96,0.90,1.92],"status":"ok"}
```

Then run geometry metrics and the eight-view CLIPScore:

```bash
python scripts/evaluate_generation.py \
  --run block3d=outputs/eval100 \
  --output-dir outputs/eval100_metrics \
  --surface-samples 8192 \
  --fscore-threshold-pct 0.01 \
  --fscore-threshold-pct 0.02 \
  --seed 0 \
  --clipscore \
  --clip-model model_weights/clip-vit-large-patch14 \
  --clip-render-nviews 8 \
  --clip-render-resolution 512 \
  --clip-render-backend pytorch3d
```

The evaluator writes `samples.jsonl`, `samples.csv`, and `summary.json`. It
centers and isotropically normalizes each mesh, samples 8,192 surface points,
and uses eight neutral-gray views for CLIPScore.

## Reference Results

Reported values on the fixed 100-object set:

| Variant | CD-L1 | NC | F@1% | F@2% |
| --- | ---: | ---: | ---: | ---: |
| M2T only | 0.0813 | 0.6664 | 0.2872 | 0.5403 |
| M2T + T2T | 0.0775 | 0.6676 | 0.3088 | 0.5507 |

The complete Block3D model reports 4.99 s mean end-to-end latency on one A100
80GB GPU, including text encoding, token generation, and mesh decoding.

## Citation

```bibtex
@article{block3d2026,
  title   = {Block3D: Efficient Text-to-3D Generation via Block-Wise Diffusion},
  author  = {Cui, Bowen and Wang, Weijie and Zhang, Zeyu and He, Yefei and
             Lin, Mingda and Zhao, Haoyu and He, Yuanyu and Chen, Donny Y. and
             Chen, Feng and Zhuang, Bohan},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

This project is distributed under the Research-Only RAIL-MS terms preserved in
`LICENSE`. Block3D is a derivative of the Cube code and released weights.
