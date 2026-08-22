import argparse
import os

import trimesh

from block3d.inference.engine import Engine
from block3d.inference.utils import normalize_bbox, select_device
from block3d.mesh_utils.postprocessing import (
    PYMESHLAB_AVAILABLE,
    create_pymeshset,
    postprocess_mesh,
    save_mesh,
)


def generate_mesh(
    engine,
    prompt,
    output_dir,
    output_name,
    resolution_base=8.0,
    disable_postprocess=False,
    top_p=None,
    bounding_box_xyz=None,
    num_diffusion_steps=4,
    sampling_strategy="block3d",
    guidance_scale=3.0,
):
    mesh_v_f = engine.t2s(
        [prompt],
        guidance_scale=guidance_scale,
        resolution_base=resolution_base,
        top_p=top_p,
        bounding_box_xyz=bounding_box_xyz,
        num_diffusion_steps=num_diffusion_steps,
        sampling_strategy=sampling_strategy,
    )
    vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
    obj_path = os.path.join(output_dir, f"{output_name}.obj")
    if PYMESHLAB_AVAILABLE:
        ms = create_pymeshset(vertices, faces)
        if not disable_postprocess:
            target_face_num = max(10000, int(faces.shape[0] * 0.1))
            print(f"Postprocessing mesh to {target_face_num} faces")
            postprocess_mesh(ms, target_face_num, obj_path)

        save_mesh(ms, obj_path)
    else:
        print(
            "WARNING: pymeshlab is not available, using trimesh to export obj and skipping optional post processing."
        )
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(obj_path)

    return obj_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block3D text-to-mesh generation")
    parser.add_argument(
        "--config-path",
        type=str,
        default="block3d/configs/block3d.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="Path to the output directory for the generated OBJ file.",
    )
    parser.add_argument(
        "--gpt-ckpt-path",
        type=str,
        required=True,
        help="Path to the main GPT checkpoint file.",
    )
    parser.add_argument(
        "--shape-ckpt-path",
        type=str,
        required=True,
        help="Path to the shape encoder/decoder checkpoint file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generating a 3D mesh",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Float < 1: keep the smallest token set with cumulative probability >= top_p. Default None is deterministic.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--bounding-box-xyz",
        nargs=3,
        type=float,
        help="Three float values for x, y, z bounding box",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--disable-postprocessing",
        help="Disable postprocessing on the mesh. This will result in a mesh with more faces.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--resolution-base",
        type=float,
        default=8.0,
        help="Resolution base for the shape decoder.",
    )
    parser.add_argument(
        "--num-diffusion-steps",
        type=int,
        default=4,
        help="Number of block diffusion denoising steps per block.",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="block3d",
        choices=("block3d",),
        help="Sampling strategy used inside block diffusion decoding.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = select_device()
    print(f"Using device: {device}")
    engine = Engine(
        args.config_path, args.gpt_ckpt_path, args.shape_ckpt_path, device=device
    )

    if args.bounding_box_xyz is not None:
        args.bounding_box_xyz = normalize_bbox(tuple(args.bounding_box_xyz))

    # Generate meshes based on input source
    obj_path = generate_mesh(
        engine=engine,
        prompt=args.prompt,
        output_dir=args.output_dir,
        output_name="output",
        resolution_base=args.resolution_base,
        disable_postprocess=args.disable_postprocessing,
        top_p=args.top_p,
        bounding_box_xyz=args.bounding_box_xyz,
        num_diffusion_steps=args.num_diffusion_steps,
        sampling_strategy=args.sampling_strategy,
        guidance_scale=args.guidance_scale,
    )
    print(f"Generated mesh for {args.prompt} at `{obj_path}`")
