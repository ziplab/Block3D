from typing import Literal, Union

import numpy as np
import torch


def _require_warp():
    try:
        import warp as wp  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "warp is required for GPU marching cubes; install it or allow the "
            "caller to fall back to CPU marching cubes."
        ) from exc
    return wp


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution_base: float,
    indexing: Literal["xy", "ij"] = "ij",
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """
    Generate a dense grid of points within a bounding box.

    Parameters:
        bbox_min (np.ndarray): The minimum coordinates of the bounding box (3D).
        bbox_max (np.ndarray): The maximum coordinates of the bounding box (3D).
        resolution_base (float): The base resolution for the grid. The number of cells along each axis will be 2^resolution_base.
        indexing (Literal["xy", "ij"], optional): The indexing convention for the grid. "xy" for Cartesian indexing, "ij" for matrix indexing. Default is "ij".
    Returns:
        tuple: A tuple containing:
            - xyz (np.ndarray): A 2D array of shape (N, 3) where N is the total number of grid points. Each row represents the (x, y, z) coordinates of a grid point.
            - grid_size (list): A list of three integers representing the number of grid points along each axis.
            - length (np.ndarray): The length of the bounding box along each axis.
    """
    length = bbox_max - bbox_min
    num_cells = np.exp2(resolution_base)
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


def marching_cubes_with_warp(
    grid_logits: torch.Tensor,
    level: float,
    device: Union[str, torch.device] = "cuda",
    max_verts: int = 3_000_000,
    max_tris: int = 3_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform the marching cubes algorithm on a 3D grid with warp support.
    Args:
        grid_logits (torch.Tensor): A 3D tensor containing the grid logits.
        level (float): The threshold level for the isosurface.
        device (Union[str, torch.device], optional): The device to perform the computation on. Defaults to "cuda".
        max_verts (int, optional): The maximum number of vertices. Defaults to 3,000,000.
        max_tris (int, optional): The maximum number of triangles. Defaults to 3,000,000.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the vertices and faces of the isosurface.
    """
    if isinstance(device, torch.device):
        device = str(device)

    assert grid_logits.ndim == 3
    wp = _require_warp()
    if "cuda" in device:
        assert wp.is_cuda_available()
    else:
        raise ValueError(
            f"Device {device} is not supported for marching_cubes_with_warp"
        )

    dim = grid_logits.shape[0]
    field = wp.from_torch(grid_logits)

    iso = wp.MarchingCubes(
        nx=dim,
        ny=dim,
        nz=dim,
        max_verts=int(max_verts),
        max_tris=int(max_tris),
        device=device,
    )
    iso.surface(field=field, threshold=level)
    vertices = iso.verts.numpy()
    faces = iso.indices.numpy().reshape(-1, 3)
    return vertices, faces
