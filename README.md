<p align="center">
  <h1 align="center">Block3D: Efficient Text-to-3D Generation via Block-Wise Diffusion</h1>
  <p align="center">
    Bowen Cui<sup>†</sup>
    ·
    <a href="https://lhmd.top">Weijie Wang<sup>†,*</sup></a>
    ·
    <a href="https://steve-zeyu-zhang.github.io">Zeyu Zhang</a>
    ·
    Yefei He
    ·
    Mingda Lin
    ·
    Haoyu Zhao
    ·
    Yuanyu He
    ·
    <a href="https://donydchen.github.io">Donny Y. Chen</a>
    ·
    Feng Chen<sup>*</sup>
    ·
    <a href="https://bohanzhuang.github.io">Bohan Zhuang</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2608.19567">Paper</a> | <a href="https://alexandertsui.github.io/block3d/">Project Page</a> | <a href="https://github.com/ziplab/Block3D">Code</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="https://alexandertsui.github.io/block3d/">
    <img src="https://alexandertsui.github.io/block3d/assets/images/qualitative_main.jpg" alt="Block3D teaser" width="100%">
  </a>
</p>

Block3D is an efficient text-to-3D generation framework that shifts the causal
dependency of discrete shape tokens from individual tokens to contiguous
blocks. It generates one block at a time, denoises all tokens in the active
block in parallel, and permits bounded token correction before a block is
committed. This preserves causal structure while substantially reducing
generation latency.

## Updates

- **2026-08-24 Update:** Release the training, inference, and evaluation code
  for Block3D.

## Method

<p align="center">
  <a href="https://alexandertsui.github.io/block3d/">
    <img src="https://alexandertsui.github.io/block3d/assets/images/pipeline_02.png" alt="Block3D pipeline" width="100%">
  </a>
</p>

Given a text prompt, frozen Cube shape and text encoders produce the condition
sequence. Block3D then generates the fixed-length discrete shape sequence from
left to right in contiguous blocks. Mask-to-token recovery and token-to-token
correction jointly update the active block, after which the block is committed
and cached. The frozen Cube shape decoder converts the completed sequence into
the output mesh.

## Installation

The reference environment is Linux with **Python 3.10+**, **PyTorch 2.2+**, and
CUDA. Four NVIDIA A100 80GB GPUs are used for the reported training runs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`pymeshlab` is optional. Blender is only needed for mesh formats that cannot be
loaded directly by `trimesh`. PyTorch3D is required by the eight-view CLIPScore
evaluation command.

## Model Zoo

Block3D is initialized from the released Cube v0.5 checkpoint. Checkpoints are
not redistributed in this repository; download them from the
[Cube v0.5 model release](https://huggingface.co/Roblox/cube3d-v0.5) and place
the files under `model_weights/`.

| Model artifact | Download |
| --- | --- |
| Cube v0.5 shape GPT initialization | [shape_gpt.safetensors](https://huggingface.co/Roblox/cube3d-v0.5/resolve/main/shape_gpt.safetensors) |
| Cube v0.5 shape tokenizer and decoder | [shape_tokenizer.safetensors](https://huggingface.co/Roblox/cube3d-v0.5/resolve/main/shape_tokenizer.safetensors) |
| CLIP ViT-L/14 text encoder | [Hugging Face Transformers](https://huggingface.co/openai/clip-vit-large-patch14) |

Expected local layout:

```text
model_weights/
  shape_gpt.safetensors
  shape_tokenizer.safetensors
  clip-vit-large-patch14/
```

## Datasets

Training uses the TRELLIS-500K paired text-mesh data. Data are not included.
Restore the source records with the following layout:

```text
data/trellis500k_train/
  source_manifest.jsonl
  pairs/
  objaverse_xl_github/
  objaverse_xl_sketchfab/
  abo/
```

Each pair record contains `shape_ids`, `bbox_xyz`, `text_input_ids`, and
`text_attention_mask`. The fixed 100-object evaluation definition is included
as `random100_evaluation_manifest.jsonl`.

## Training

### Preparation

The reported runs use a 300K training split after excluding the fixed 100-object
evaluation set:

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

### Block3D

```bash
torchrun --standalone --nproc_per_node=4 \
  -m block3d.train_block_diffusion \
  --train-config-path block3d/configs/train_block3d.yaml
```

### M2T-only ablation

```bash
torchrun --standalone --nproc_per_node=4 \
  -m block3d.train_block_diffusion \
  --train-config-path block3d/configs/train_block3d_m2t_only.yaml
```

Both runs use 35K optimizer steps, global batch size 40, bfloat16 precision,
seed 42, AdamW, and learning rate `1e-4`. The final generator checkpoint is
written to `runs/<run>/checkpoints/gpt_final.safetensors`.

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

The output mesh is `outputs/chair/output.obj`. The reported text-only protocol
does not pass `--bounding-box-xyz`. Batch-generation orchestration is not part
of this release.

## Evaluation

Create `outputs/eval100/samples.jsonl` with one record per generated mesh:

```json
{"sample_id":"0000","prompt_text":"...","reference_mesh_path":"...","generated_mesh_path":"0000/generated.obj","bbox_xyz":[0.96,0.90,1.92],"status":"ok"}
```

Run geometry metrics and the eight-view CLIPScore:

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

## Results

Reported values on the fixed 100-object evaluation set:

| Variant | CD-L1 | NC | F@1% | F@2% |
| --- | ---: | ---: | ---: | ---: |
| M2T only | 0.0813 | 0.6664 | 0.2872 | 0.5403 |
| M2T + T2T | 0.0775 | 0.6676 | 0.3088 | 0.5507 |

The complete Block3D model reports 4.99 s mean end-to-end latency on one A100
80GB GPU, including text encoding, token generation, and mesh decoding.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{cui2026block3d,
  title   = {Block3D: Efficient Text-to-3D Generation via Block-Wise Diffusion},
  author  = {Cui, Bowen and Wang, Weijie and Zhang, Zeyu and He, Yefei and
             Lin, Mingda and Zhao, Haoyu and He, Yuanyu and Chen, Donny Y. and
             Chen, Feng and Zhuang, Bohan},
  journal = {arXiv preprint arXiv:2608.19567},
  year    = {2026}
}
```

## Contact

Please use the issue tracker of this repository for questions, bug reports, or
reproduction issues.

## Acknowledgements

This project builds on the released Cube shape tokenizer, decoder, and GPT
backbone, and uses TRELLIS-500K and LLaDA2.1 as described in the paper. We thank
the original authors for making their work available.

## License

This project is distributed under the Research-Only RAIL-MS terms preserved in
`LICENSE`. Block3D is a derivative work based on Cube3D-v0.1 and its released
weights; please review the license before use.
