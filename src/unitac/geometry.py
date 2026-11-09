"""Cold-path geometry conversion and deterministic surface sampling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from unisim.ray_query import RayCollisionDescription, RayGeometryState, RayGeomType


def body_meshes(
    collision: RayCollisionDescription,
    body_names: Sequence[str],
    *,
    geom_names: Sequence[str] | None = None,
    geometry_state: RayGeometryState | None = None,
) -> tuple[np.ndarray, ...]:
    """Convert selected public collision geoms to body-local triangles.

    Per-environment geometry snapshots are accepted only when the selected
    geometry is identical across every row; HydroShear materializes one static
    mesh set and cannot represent row-varying geometry.
    """
    body_ids = {name: index for index, name in enumerate(collision.scene.body_names)}
    mesh_by_id = {index: mesh for index, mesh in enumerate(collision.meshes)}
    if geom_names is None:
        selected_geom_names = None
    else:
        selected_geom_names = set(geom_names)
    if geometry_state is not None:
        if not isinstance(geometry_state, RayGeometryState):
            raise TypeError("geometry_state must be a RayGeometryState")
        if geometry_state.geom_names != collision.geom_names:
            raise ValueError("geometry_state geom_names must match collision geom_names in order")
    result = []
    for body_name in body_names:
        body_id = body_ids[body_name]
        parts = []
        for geom_id, geom_body in enumerate(collision.scene.geom_body_ids):
            if int(geom_body) != body_id:
                continue
            geom_name = collision.geom_names[geom_id]
            if selected_geom_names is not None and geom_name not in selected_geom_names:
                continue
            if geometry_state is None:
                geom_pos = collision.scene.geom_local_pos[geom_id]
                geom_quat = collision.scene.geom_local_quat[geom_id]
                mesh_id = int(collision.geom_mesh_ids[geom_id])
                enabled = True
            else:
                geom_pos = _uniform_geometry_row(
                    geometry_state.geom_local_pos[:, geom_id], geom_name, "local pose"
                )
                geom_quat = _uniform_geometry_row(
                    geometry_state.geom_local_quat[:, geom_id], geom_name, "local pose"
                )
                mesh_id = int(
                    _uniform_geometry_row(
                        geometry_state.geom_mesh_ids[:, geom_id], geom_name, "mesh identity"
                    )
                )
                enabled = bool(
                    _uniform_geometry_row(
                        geometry_state.geom_enabled[:, geom_id], geom_name, "enabled state"
                    )
                )
            if not enabled:
                continue
            if mesh_id >= 0:
                try:
                    mesh = mesh_by_id[mesh_id]
                except KeyError as error:
                    raise ValueError(
                        f"geometry_state mesh id {mesh_id} for {geom_name!r} is absent"
                    ) from error
                # RayMeshDescription points already include mesh asset scaling
                # and are expressed in the compiled geom-local frame. MuJoCo
                # ignores geom_size for mesh geoms; multiplying it here applies
                # a second, invalid scale.
                triangles = mesh.points[mesh.indices.reshape(-1, 3)]
            else:
                if geometry_state is None:
                    geom_size = collision.scene.geom_sizes[geom_id]
                else:
                    geom_size = _uniform_geometry_row(
                        geometry_state.geom_sizes[:, geom_id], geom_name, "size"
                    )
                triangles = _primitive_triangles(
                    collision.scene.geom_types[geom_id],
                    geom_size,
                )
            triangles = _rotate(triangles, geom_quat)
            triangles += geom_pos
            parts.append(triangles.astype(np.float32, copy=False))
        if not parts:
            raise ValueError(f"tactile object body {body_name!r} has no collision geometry")
        result.append(np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32))
    return tuple(result)


def _uniform_geometry_row(
    values: np.ndarray, geom_name: str, field: str
) -> np.ndarray | np.generic:
    if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
        raise NotImplementedError(
            f"HydroShear does not support per-environment geometry {field} variation "
            f"for geom {geom_name!r}; rebuild requires one shared geometry"
        )
    return values[0]


def sample_object_surfaces(meshes: Sequence[np.ndarray], count: int) -> tuple[np.ndarray, ...]:
    """Allocate deterministic area-weighted samples across object bodies."""
    if len(meshes) > count:
        raise ValueError("surface_points must be at least the number of perceived objects")
    areas = np.asarray(
        [
            np.linalg.norm(np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0]), axis=1).sum()
            for mesh in meshes
        ],
        dtype=np.float64,
    )
    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ValueError("each perceived object must have positive finite surface area")
    quotas = (count - len(meshes)) * areas / areas.sum()
    counts = 1 + np.floor(quotas).astype(np.intp)
    remainder = count - int(counts.sum())
    if remainder:
        order = np.argsort(quotas - np.floor(quotas))[::-1]
        counts[order[:remainder]] += 1
    return tuple(
        sample_surface_points(mesh, int(point_count), seed=index)
        for index, (mesh, point_count) in enumerate(zip(meshes, counts, strict=True))
    )


def sample_surface_points(triangles: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """Sample mesh surface points proportionally to triangle area."""
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("triangles must have positive finite surface area")
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(triangles), size=count, p=areas / total_area)
    uv = rng.random((count, 2), dtype=np.float32)
    root = np.sqrt(uv[:, :1])
    barycentric = np.concatenate((1.0 - root, root * (1.0 - uv[:, 1:]), root * uv[:, 1:]), axis=1)
    return np.einsum("ni,nij->nj", barycentric, triangles[face_ids]).astype(np.float32, copy=False)


def _rotate(points: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    xyz = quaternion[1:]
    tangent = 2.0 * np.cross(xyz, points)
    return points + quaternion[:1] * tangent + np.cross(xyz, tangent)


def _box(half: np.ndarray) -> np.ndarray:
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
    faces = np.asarray(
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
        dtype=np.intp,
    )
    return vertices[faces]


def _uv_sphere(radii: np.ndarray, *, segments: int = 32, rings: int = 16) -> np.ndarray:
    angles = np.arange(segments, dtype=np.float64) * (2.0 * np.pi / segments)
    vertices: list[list[float]] = [[0.0, 0.0, float(radii[2])]]
    for ring in range(1, rings):
        polar = np.pi * ring / rings
        vertices.extend(
            [
                float(radii[0] * np.sin(polar) * np.cos(angle)),
                float(radii[1] * np.sin(polar) * np.sin(angle)),
                float(radii[2] * np.cos(polar)),
            ]
            for angle in angles
        )
    bottom = len(vertices)
    vertices.append([0.0, 0.0, -float(radii[2])])
    faces: list[tuple[int, int, int]] = []
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append((0, 1 + nxt, 1 + segment))
        for ring in range(rings - 2):
            low, high = 1 + ring * segments, 1 + (ring + 1) * segments
            faces.extend(
                (
                    (low + segment, low + nxt, high + segment),
                    (low + nxt, high + nxt, high + segment),
                )
            )
        last = 1 + (rings - 2) * segments
        faces.append((bottom, last + segment, last + nxt))
    return np.asarray(vertices, dtype=np.float32)[np.asarray(faces, dtype=np.intp)]


def _cylinder(radius: float, half_length: float, segments: int = 32) -> np.ndarray:
    angles = np.arange(segments, dtype=np.float64) * (2.0 * np.pi / segments)
    circle = np.stack((radius * np.cos(angles), radius * np.sin(angles)), axis=1)
    vertices = np.concatenate(
        (
            np.column_stack((circle, np.full(segments, -half_length))),
            np.column_stack((circle, np.full(segments, half_length))),
            [[0.0, 0.0, -half_length], [0.0, 0.0, half_length]],
        ),
        axis=0,
    )
    bottom, top = 2 * segments, 2 * segments + 1
    faces: list[tuple[int, int, int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.extend(((index, nxt, segments + index), (nxt, segments + nxt, segments + index)))
        faces.extend(((bottom, nxt, index), (top, segments + index, segments + nxt)))
    return vertices[np.asarray(faces, dtype=np.intp)].astype(np.float32)


def _primitive_triangles(geom_type: RayGeomType, size: np.ndarray) -> np.ndarray:
    if geom_type == RayGeomType.BOX:
        return _box(size)
    if geom_type == RayGeomType.SPHERE:
        return _uv_sphere(np.repeat(size[0], 3))
    if geom_type == RayGeomType.ELLIPSOID:
        return _uv_sphere(size)
    if geom_type == RayGeomType.CYLINDER:
        return _cylinder(float(size[0]), float(size[1]))
    if geom_type == RayGeomType.CAPSULE:
        radius, half_length = float(size[0]), float(size[1])
        if half_length == 0.0:
            return _uv_sphere(np.repeat(radius, 3))
        segments, arc_steps = 32, 8
        angles = np.arange(segments) * (2.0 * np.pi / segments)
        ring_params = [
            (-half_length, -np.pi / 2 + step * np.pi / (2 * arc_steps))
            for step in range(1, arc_steps + 1)
        ]
        ring_params.append((half_length, 0.0))
        ring_params.extend(
            (half_length, step * np.pi / (2 * arc_steps)) for step in range(1, arc_steps)
        )
        rings = [
            np.stack(
                (
                    radius * np.cos(polar) * np.cos(angles),
                    radius * np.cos(polar) * np.sin(angles),
                    np.full(segments, center + radius * np.sin(polar)),
                ),
                axis=1,
            )
            for center, polar in ring_params
        ]
        vertices = np.concatenate(rings, axis=0)
        faces: list[tuple[int, int, int]] = []
        bottom_pole = len(vertices)
        top_pole = bottom_pole + 1
        vertices = np.concatenate(
            (
                vertices,
                [[0.0, 0.0, -half_length - radius], [0.0, 0.0, half_length + radius]],
            )
        )
        faces.extend((bottom_pole, (index + 1) % segments, index) for index in range(segments))
        for ring in range(len(rings) - 1):
            for index in range(segments):
                nxt = (index + 1) % segments
                a, b = ring * segments + index, ring * segments + nxt
                c, d = (ring + 1) * segments + index, (ring + 1) * segments + nxt
                faces.extend(((a, b, c), (b, d, c)))
        last_ring = (len(rings) - 1) * segments
        faces.extend(
            (top_pole, last_ring + index, last_ring + (index + 1) % segments)
            for index in range(segments)
        )
        return vertices[np.asarray(faces, dtype=np.intp)].astype(np.float32)
    raise ValueError(
        f"HydroShear requires closed collision geometry; unsupported type {geom_type!r}"
    )
