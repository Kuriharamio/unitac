"""Synthetic closed meshes and detached UniSim tactile scenes."""

from __future__ import annotations

import numpy as np
from unisim.ray_query import (
    RayCollisionDescription,
    RayGeomType,
    RayMeshDescription,
    RaySceneDescription,
    RaySiteFrameDescription,
)
from unisim.tactile import TactileSceneDescription

IDENTITY = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
BOX_FACES = np.asarray(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int32,
)


def box_triangles(half: tuple[float, float, float]) -> np.ndarray:
    hx, hy, hz = half
    vertices = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    return vertices[BOX_FACES]


def box_vertices(half: tuple[float, float, float]) -> np.ndarray:
    hx, hy, hz = half
    return np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )


def make_scene(object_names: tuple[str, ...] = ("object",)) -> TactileSceneDescription:
    body_names = (*object_names, "pad0", "pad1")
    meshes = tuple(
        RayMeshDescription(
            asset_id=f"synthetic:{name}",
            points=box_vertices((0.005, 0.005, 0.004)),
            indices=BOX_FACES.reshape(-1),
        )
        for name in object_names
    )
    collision = RayCollisionDescription(
        scene=RaySceneDescription(
            num_bodies=len(body_names),
            geom_types=tuple(RayGeomType.MESH for _ in object_names),
            geom_sizes=np.zeros((len(object_names), 3)),
            geom_local_pos=np.zeros((len(object_names), 3)),
            geom_local_quat=np.tile(IDENTITY, (len(object_names), 1)),
            geom_body_ids=np.arange(len(object_names), dtype=np.int32),
            body_names=body_names,
        ),
        geom_names=tuple(f"{name}_mesh" for name in object_names),
        meshes=meshes,
        geom_mesh_ids=np.arange(len(object_names), dtype=np.int32),
        body_parent_ids=np.full(len(body_names), -1, dtype=np.int32),
    )
    frames = tuple(
        RaySiteFrameDescription(f"sensor{index}", f"pad{index}", np.zeros(3), IDENTITY)
        for index in range(2)
    )
    return TactileSceneDescription(collision, object_names, frames)


def hydro_config(gel_mesh: np.ndarray, *, surface_points: int = 2048) -> dict:
    return {
        "gel_mesh": gel_mesh,
        "surface_points": surface_points,
        "output_grid": {
            "rows": 9,
            "cols": 7,
            "spacing_mm": [2.145, 2.153125],
            "center_offset_mm": [-0.3575, -0.325],
            "pixel_scale_mm": 0.065,
        },
        "solver": {"use_fft_dilation": False, "use_matmul_shear": True},
        "device": "cuda:0",
    }


def full_body_poses(
    num_envs: int,
    object_positions: np.ndarray,
    object_quaternions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    object_count = object_positions.shape[1]
    positions = np.zeros((num_envs, object_count + 2, 3), dtype=np.float32)
    quaternions = np.tile(IDENTITY.astype(np.float32), (num_envs, object_count + 2, 1))
    positions[:, :object_count] = object_positions
    if object_quaternions is not None:
        quaternions[:, :object_count] = object_quaternions
    positions[:, object_count, 0] = -0.004
    positions[:, object_count + 1, 0] = 0.004
    return positions, quaternions
