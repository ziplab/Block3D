import argparse
import sys
from pathlib import Path

import bpy


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Expected Blender arguments after --")
    parser = argparse.ArgumentParser(description="Convert a mesh asset to OBJ via Blender.")
    parser.add_argument("--input", required=True, help="Path to the source mesh asset.")
    parser.add_argument("--output", required=True, help="Path to the output .obj file.")
    return parser.parse_args(argv[argv.index("--") + 1 :])


def _clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_asset(input_path: Path) -> None:
    suffix = input_path.suffix.lower()
    if suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(input_path))
        return

    _clear_scene()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(input_path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(input_path))
    elif suffix == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=str(input_path))
        except Exception:
            bpy.ops.import_scene.obj(filepath=str(input_path))
    elif suffix == ".stl":
        bpy.ops.import_mesh.stl(filepath=str(input_path))
    elif suffix == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(input_path))
    elif suffix == ".dae":
        bpy.ops.wm.collada_import(filepath=str(input_path))
    else:
        raise RuntimeError(f"Unsupported Blender conversion suffix: {suffix}")


def _select_mesh_objects() -> list[object]:
    bpy.ops.object.select_all(action="DESELECT")
    mesh_objects = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        obj.hide_set(False)
        obj.hide_select = False
        obj.hide_viewport = False
        obj.select_set(True)
        mesh_objects.append(obj)
    if mesh_objects:
        bpy.context.view_layer.objects.active = mesh_objects[0]
    return mesh_objects


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _import_asset(input_path)
    mesh_objects = _select_mesh_objects()
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects found in {input_path}")

    try:
        bpy.ops.wm.obj_export(
            filepath=str(output_path),
            export_selected_objects=True,
            export_materials=False,
        )
    except Exception:
        bpy.ops.export_scene.obj(
            filepath=str(output_path),
            use_selection=True,
            use_materials=False,
            keep_vertex_order=True,
        )


if __name__ == "__main__":
    main()
