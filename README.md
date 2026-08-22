# Block3D: Anonymous Supplementary Code

This archive contains the paper-aligned training, text-to-mesh inference, and
evaluation implementation for Block3D. Block3D converts Cube's 1,024-code shape
prior into a block-causal editable decoder with mask-to-token (M2T) recovery and
token-to-token (T2T) correction.

The archive is deliberately self-contained as supplementary source: it contains
no external repository links, author identities, pretrained weights, third-party
datasets, generated meshes, or machine-specific paths. `LICENSE` is retained
because this implementation is a derivative of Cube and must preserve the
upstream Research-Only RAIL-MS terms.

## Archive Contents

```text
block3d/
  configs/                    Paper configurations for the main and ablation models
  inference/                  Block3D text-to-shape decoding
  mesh_utils/                 OBJ export and optional mesh postprocessing
  model/                      Shape tokenizer and token generator modules
  training/                   Dataset loading, objectives, and trainer
  benchmarking.py             Fixed-manifest loading and latency summaries
  evaluation.py               Geometry and eight-view CLIPScore implementation
  generate.py                 Single-prompt text-to-mesh entry point
  train_block_diffusion.py    Distributed training entry point
scripts/
  evaluate_generation.py      Scores already-generated meshes
  materialize_ablation_configs.py
                               Writes the B/T ablation configuration matrix
  prepare_block_diffusion_eval_split.py
                               Excludes eval100 and selects the 300K train subset
random100_evaluation_manifest.jsonl
                               Ordered 100-object evaluation definition
LICENSE
pyproject.toml
README.md
```

The package intentionally does not include a batch-generation scheduler. The
single-prompt command below is the generation interface; the fixed JSONL manifest
provides the ordered prompts for an external job scheduler. This also keeps
cluster-specific launch and recovery scaffolding out of the submission.

## Environment

The reference environment is Linux, Python 3.10, PyTorch 2.2.2 or later with
CUDA, and four NVIDIA A100 80GB GPUs for training. The CPU performs data loading,
mesh import, and process orchestration; all latency rows are measured on the same
host, and the paper does not report a CPU-throughput comparison. From the archive
root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`pymeshlab` is optional and can be installed with `python -m pip install -e
'.[meshlab]'`. Without it, inference exports through `trimesh` and skips optional
mesh postprocessing. Blender is needed only when a source mesh cannot be loaded
directly. The paper-aligned CLIP rendering command uses PyTorch3D; install a
PyTorch3D build compatible with the local PyTorch and CUDA versions before that
evaluation.

Each training launch writes `training_run.json`. In addition to the resolved
command and git revision, this file records the operating-system platform, CPU
model and logical core count, host memory, Python and PyTorch versions, CUDA
runtime version, GPU names, and GPU memory. This avoids treating the machine on
which the archive is inspected as the environment of a reported run.

## Required External Artifacts

The source archive alone is not sufficient to reproduce training or inference.
The following upstream artifacts must first be restored locally:

```text
model_weights/
  shape_gpt.safetensors
  shape_tokenizer.safetensors
  clip-vit-large-patch14/
data/
  trellis500k_train/
    source_manifest.jsonl
    pairs/
    objaverse_xl_github/
    objaverse_xl_sketchfab/
    abo/
  splits/                     Created by the split command below
```

The two safetensors files are the released Cube initialization from release
identifier `Roblox/cube3d-v0.5`. The CLIP directory must be a complete local,
Transformers-compatible CLIP ViT-L/14 model directory. The data directory must
contain the referenced TRELLIS-500K snapshot and its pre-tokenized pair records.
These large third-party artifacts are not redistributed in this supplementary
archive.

Training from the paper's initialization therefore requires all three model
artifacts plus the training data. Once they are restored, the provided entry
point and configurations train the 35K-step Block3D generator. The VQ shape
encoder, codebook, CLIP text encoder, and shape decoder remain frozen; only the
text-to-shape generator is optimized. Random initialization would be a different
experiment and is not the reported protocol.

## Data Records

Each row of `data/trellis500k_train/source_manifest.jsonl` is a JSON object. A
pre-tokenized record uses this schema:

```json
{"mesh_path":"objaverse_xl_github/path/to/asset.glb","text":"a wooden chair","pair_path":"pairs/item.pt"}
```

Paths are relative to `data/trellis500k_train`. Each `.pt` pair record must be a
dictionary containing `shape_ids`, `bbox_xyz`, `text_input_ids`, and
`text_attention_mask`. The stored bounding box is retained for geometry metadata
but is not passed to the generator during the reported text-only training.

`random100_evaluation_manifest.jsonl` contains exactly 100 ordered records with
stable IDs `0000` through `0099`, 100 distinct prompts, and 100 distinct target
mesh paths. Its source composition is 53 Objaverse-XL Sketchfab objects, 46
Objaverse-XL GitHub objects, and one ABO object, all referenced through the
TRELLIS-500K snapshot. The manifest identifies third-party targets rather than
redistributing them. `generated_mesh_relpath`, `generated_mesh_sha256`, and
`generation_status` are provenance for the archived experiment; fresh generated
meshes are supplied separately through the evaluation `samples.jsonl` described
below.

## Recreate the Paper Split

Exclude the fixed 100 evaluation objects and deterministically select the 300K
training objects with seed 42:

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

The script stops if the fixed manifest does not contain exactly 100 records, an
evaluation object cannot be matched to the source pool, the exclusion count is
not 100, fewer than 300K candidates remain, or any train/eval overlap is found.
It produces:

```text
data/splits/
  train_manifest.jsonl
  val_manifest.jsonl                         Empty in the paper protocol
  eval100_manifest.jsonl
  excluded_eval100_source_records.jsonl
  split_summary.json
```

`split_summary.json` records source, excluded, candidate, selected, validation,
and overlap counts. The two supplied training YAML files read
`data/splits/train_manifest.jsonl` and perform no validation-based model
selection; the final 35K-step checkpoint is evaluated.

## Paper Configuration

The main model is defined by `block3d/configs/block3d.yaml` and
`block3d/configs/train_block3d.yaml`:

| Setting | Value |
| --- | --- |
| Shape sequence length / vocabulary | 1024 / 16384 |
| Generator | 23 layers, 12 heads, width 1536 |
| Block size / denoising steps per block | 64 / 4 |
| M2T / T2T confidence thresholds | 0.95 / 0.9 |
| CFG scale / decoding | 3.0 / deterministic argmax |
| Training corruption range | [0.45, 0.95] |
| T2T stream probability / rollout steps | 0.5 / 1 |
| Condition dropout | 0.1 |
| Optimizer | AdamW, lr 1e-4, betas (0.9, 0.95), weight decay 0 |
| Warm-up / gradient clipping | 100 steps / 1.0 |
| Steps / global batch | 35000 / 40 (10 per GPU) |
| Precision / training seed | bfloat16 / 42 |

The architecture retains Cube's optional bbox projection so that the released
initialization loads exactly, but the commands below do not append a bbox token.
Both reported training and inference use the 77-token text condition only.

`block3d/configs/block3d_m2t_only.yaml` defines the independently trained
M2T-only comparison by setting `enable_t2t: false`, `t2t_training_ratio: 0`, and
`mtf_rollout_steps: 0`. The main configuration uses the one-step model rollout
reported in the paper.

Materialize the independently trained block-size and denoising-step
configurations used in Tables 3 and 4 of the paper:

```bash
python scripts/materialize_ablation_configs.py
```

This writes model/training pairs for `B` in 32, 96, 128, and 256 and for `T` in
8, 12, and 20 under `block3d/configs/ablations/`. The main configuration supplies
the `B=64, T=4` row. Every generated training file retains the 35K-step budget,
optimizer, batch size, corruption interval, condition dropout, and seed of the
main run while changing only the named block-size or step setting.

## Training

Run the main experiment on four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m block3d.train_block_diffusion \
  --train-config-path block3d/configs/train_block3d.yaml
```

Run the M2T-only comparison:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m block3d.train_block_diffusion \
  --train-config-path block3d/configs/train_block3d_m2t_only.yaml
```

The main run writes the following structure; the ablation uses
`runs/block3d_m2t_only/` with the same layout:

```text
runs/block3d/
  resolved_train_config.yaml
  resolved_config.yaml
  training_run.json
  dataset_summary.json
  manifests/
    train_entries.jsonl
    val_entries.jsonl
  logs/
    train.jsonl
    tensorboard/
  checkpoints/
    gpt_step_00005000.safetensors
    gpt_step_00010000.safetensors
    ...
    gpt_step_00035000.safetensors
    gpt_final.safetensors
```

## Single-Prompt Inference

Generate from text only with the final checkpoint:

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

Expected output:

```text
outputs/chair/
  output.obj
```

Do not pass `--bounding-box-xyz` for the reported protocol. Omitting `--top-p`
uses deterministic argmax decoding, and `--disable-postprocessing` exports the
native decoded geometry without optional PyMeshLab cleanup. Repeat this command
once for each ordered prompt in `random100_evaluation_manifest.jsonl`, placing
each output under its corresponding sample ID. Batch orchestration is
intentionally not part of this archive.

## Evaluation Input

Before evaluation, create `outputs/eval100/samples.jsonl` with one row per
generated mesh in the same `0000` to `0099` order:

```json
{"sample_id":"0000","prompt_text":"A blocky car ...","reference_mesh_path":"data/trellis500k_train/objaverse_xl_github/path/to/target.gltf","generated_mesh_path":"0000/generated.obj","bbox_xyz":[0.96,0.90,1.92],"status":"ok"}
```

Required fields are `sample_id`, `prompt_text`, `reference_mesh_path`, and
`generated_mesh_path`. `bbox_xyz` is optional; when present it determines the
relative F-score thresholds and is never passed to generation. `status` should
be `ok` for a valid generation or `failed`/`missing` otherwise. Relative paths
are resolved from the JSONL location and known local data roots.

A typical input tree is:

```text
outputs/eval100/
  samples.jsonl
  0000/generated.obj
  0001/generated.obj
  ...
  0099/generated.obj
```

## Geometry and CLIPScore

Run the complete paper evaluation with 8,192 surface samples, NumPy base seed 0,
the F-score threshold at 1% of the stored target bbox diagonal, and eight-view
CLIPScore. Sample index `i` uses surface-sampling seed `i`, preserving fixed but
distinct samples across the ordered manifest:

```bash
python scripts/evaluate_generation.py \
  --run block3d=outputs/eval100 \
  --output-dir outputs/eval100_metrics \
  --surface-samples 8192 \
  --fscore-threshold-pct 0.01 \
  --seed 0 \
  --clipscore \
  --clip-model model_weights/clip-vit-large-patch14 \
  --clip-render-nviews 8 \
  --clip-render-resolution 512 \
  --clip-render-backend pytorch3d
```

For each mesh, the PyTorch3D backend uses a neutral-gray material, white
background, fixed lighting, camera elevation 30 degrees, radius 2, field of view
40 degrees, and azimuths separated by 45 degrees. CLIPScore is the mean over the
eight positive cosine scores multiplied by 100. Geometry evaluation centers and
isotropically rescales each mesh, samples by triangle area, performs no rotational
alignment or ICP, and records failures rather than regenerating outputs.

Expected evaluation output:

```text
outputs/eval100_metrics/
  samples.jsonl
  samples.csv
  summary.json
  sample_artifacts/block3d/<sample-id>/renders/generated_mvclip/
```

## Reference Results

The paper reports the following single-seed results on the ordered 100-object
evaluation set:

| Variant | CD-L1 (lower) | NC (higher) | F@1% (higher) |
| --- | ---: | ---: | ---: |
| M2T only | 0.0813 | 0.6664 | 0.2872 |
| M2T + T2T | 0.0775 | 0.6676 | 0.3088 |

The complete Block3D model records mean end-to-end latency of 4.99 seconds at
batch size one on one NVIDIA A100 80GB GPU. Timing includes text-condition
encoding, shape-code generation, and mesh decoding. Models remain resident on
the device; untimed warm-up runs are completed; CUDA is synchronized immediately
before and after every generation; model loading, checkpoint transfer,
visualization, and disk I/O are excluded.

These values are reference outcomes, not bitwise guarantees. Hardware, CUDA and
library versions, mesh import behavior, and floating-point kernels can cause
small deviations. The paper reports one training seed and one generation per
prompt; it does not claim seed variance, confidence intervals, or sampling
diversity.

## License and Upstream Attribution

Block3D is a derivative of the Cube code and released weights. The header in
`LICENSE` identifies Block3D as the derivative work; the remainder preserves the
complete upstream Research-Only RAIL-MS license for Cube3d-v0.1 and related
inference code. The v0.1 label is the upstream licensed-artifact name, while the
paper initialization uses the later Cube v0.5 weight release identified above.
