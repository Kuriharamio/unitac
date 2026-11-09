from __future__ import annotations

import numpy as np
import pytest
from unisim.ray_query import (
    RayCollisionDescription,
    RayGeometryState,
    RayGeomType,
    RayMeshDescription,
    RaySceneDescription,
)

from unitac.geometry import body_meshes


def _collision(
    geom_types: tuple[RayGeomType, ...],
    geom_sizes: np.ndarray,
    geom_local_pos: np.ndarray,
    geom_local_quat: np.ndarray,
    geom_body_ids: np.ndarray,
    *,
    meshes: tuple[RayMeshDescription, ...] = (),
    geom_mesh_ids: np.ndarray | None = None,
    geom_names: tuple[str, ...] | None = None,
) -> RayCollisionDescription:
    body_names = ("object", "pad")
    scene = RaySceneDescription(
        num_bodies=len(body_names),
        geom_types=geom_types,
        geom_sizes=geom_sizes,
        geom_local_pos=geom_local_pos,
        geom_local_quat=geom_local_quat,
        geom_body_ids=geom_body_ids,
        body_names=body_names,
    )
    return RayCollisionDescription(
        scene=scene,
        geom_names=geom_names or tuple(f"geom_{index}" for index in range(len(geom_types))),
        meshes=meshes,
        geom_mesh_ids=geom_mesh_ids,
        body_parent_ids=np.asarray([-1, 0], dtype=np.int32),
    )


def _state(collision: RayCollisionDescription, **updates) -> RayGeometryState:
    scene = collision.scene
    geom_count = scene.num_geoms
    values = {
        "num_envs": 2,
        "geom_names": collision.geom_names,
        "geom_sizes": np.tile(scene.geom_sizes[None], (2, 1, 1)),
        "geom_local_pos": np.tile(scene.geom_local_pos[None], (2, 1, 1)),
        "geom_local_quat": np.tile(scene.geom_local_quat[None], (2, 1, 1)),
        "geom_enabled": np.ones((2, geom_count), dtype=np.bool_),
        "geom_mesh_ids": np.tile(collision.geom_mesh_ids[None], (2, 1)),
    }
    values.update(updates)
    return RayGeometryState(**values)


def test_body_meshes_handles_primitive_types_and_multiple_geoms_per_body():
    collision = _collision(
        (RayGeomType.BOX, RayGeomType.CYLINDER, RayGeomType.CAPSULE, RayGeomType.BOX),
        np.asarray([[1.0, 2.0, 3.0], [0.5, 0.75, 0.0], [0.25, 0.5, 0.0], [1, 1, 1]]),
        np.asarray([[0, 0, 0], [4, 0, 0], [0, 4, 0], [100, 0, 0]], dtype=np.float64),
        np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
        np.asarray([0, 0, 0, 1]),
        geom_names=("box", "cylinder", "capsule", "pad_geom"),
    )

    all_parts = body_meshes(collision, ("object",))[0]
    capsule_only = body_meshes(collision, ("object",), geom_names=("capsule",))[0]

    assert all_parts.shape == (12 + 128 + 1024, 3, 3)
    assert capsule_only.shape == (1024, 3, 3)
    areas = np.linalg.norm(
        np.cross(capsule_only[:, 1] - capsule_only[:, 0], capsule_only[:, 2] - capsule_only[:, 0]),
        axis=1,
    )
    assert np.all(areas > 0.0)
    assert not np.any(np.isclose(all_parts[..., 0], 100.0))


def test_body_meshes_uses_compiled_mesh_scale_and_geom_local_pose():
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.asarray([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=np.int32)
    mesh = RayMeshDescription("tetra", points, faces)
    half_turn_z = np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    collision = _collision(
        (RayGeomType.MESH,),
        np.asarray([[2.0, 3.0, 4.0]]),
        np.asarray([[5.0, 6.0, 7.0]]),
        half_turn_z[None],
        np.asarray([0]),
        meshes=(mesh,),
        geom_mesh_ids=np.asarray([0]),
    )

    triangles = body_meshes(collision, ("object",))[0]

    assert np.any(np.all(np.isclose(triangles, [5.0, 7.0, 7.0]), axis=-1))
    assert np.any(np.all(np.isclose(triangles, [4.0, 6.0, 7.0]), axis=-1))


def test_original_gelsight_collision_mesh_is_a_closed_512_face_fixture():
    from pathlib import Path

    from unitac.hydroshear.config import load_triangle_mesh

    path = Path(__file__).parent / "fixtures" / "gsmini_elastomer_collision_512.obj"
    triangles = load_triangle_mesh(path)
    assert triangles.shape == (512, 3, 3)
    np.testing.assert_allclose(
        [triangles.min(), triangles.max()],
        [-0.072375, 0.009875],
        atol=1e-6,
    )


def test_body_meshes_uses_uniform_geometry_snapshot_and_rejects_row_variation():
    collision = _collision(
        (RayGeomType.BOX,),
        np.asarray([[1.0, 1.0, 1.0]]),
        np.zeros((1, 3)),
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        np.asarray([0]),
    )
    sizes = np.tile(collision.scene.geom_sizes[None], (2, 1, 1)) * 2.0
    triangles = body_meshes(
        collision,
        ("object",),
        geometry_state=_state(collision, geom_sizes=sizes),
    )[0]
    assert triangles.min() == -2.0
    assert triangles.max() == 2.0

    positions = np.tile(collision.scene.geom_local_pos[None], (2, 1, 1))
    positions[1, 0, 0] = 0.1
    with pytest.raises(NotImplementedError, match="per-environment geometry local pose variation"):
        body_meshes(
            collision,
            ("object",),
            geometry_state=_state(collision, geom_local_pos=positions),
        )


def test_body_meshes_rejects_per_environment_enable_and_mesh_identity_changes():
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.asarray([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=np.int32)
    collision = _collision(
        (RayGeomType.MESH,),
        np.ones((1, 3)),
        np.zeros((1, 3)),
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        np.asarray([0]),
        meshes=(RayMeshDescription("tetra", points, faces),),
        geom_mesh_ids=np.asarray([0]),
    )
    enabled = np.ones((2, 1), dtype=np.bool_)
    enabled[1, 0] = False
    with pytest.raises(NotImplementedError, match="enabled state variation"):
        body_meshes(collision, ("object",), geometry_state=_state(collision, geom_enabled=enabled))

    mesh_ids = np.zeros((2, 1), dtype=np.int32)
    mesh_ids[1, 0] = -1
    with pytest.raises(NotImplementedError, match="mesh identity variation"):
        body_meshes(
            collision,
            ("object",),
            geometry_state=_state(collision, geom_mesh_ids=mesh_ids),
        )
