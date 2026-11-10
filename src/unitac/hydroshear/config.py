"""HydroShear-only model configuration and user-supplied gel geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .depth import CAMERA_H, CAMERA_W

_SOLVER_DEFAULTS: dict[str, Any] = {
    "mu": 0.5,
    "lambda_d": 20_000.0,
    "lambda_s": 10_800.0,
    # Preserve HydroShear's empirical gains; these are not SI displacement conversions.
    "shear_scale": 1000.0 / 0.065,
    "dilate_scale": 1000.0 / 0.065,
    "normal_axis": 2,
    "K": 1.0,
    "E": 1.0,
    "A": 1.0,
    "reverse_z": False,
    "use_fft_dilation": False,
    "fft_dilation_min_pts": 256,
    "fft_dilation_grid_shape": (9, 7),
    "use_matmul_shear": True,
}


@dataclass(frozen=True)
class HydroShearConfig:
    """Validated HydroShear setup; lengths are metres except grid spacing in mm."""

    gel_triangles: np.ndarray
    surface_points: int
    rows: int
    cols: int
    spacing_mm: tuple[float, float]
    center_offset_mm: tuple[float, float]
    marker_x_mm: tuple[float, ...]
    marker_y_mm: tuple[float, ...]
    pixel_scale_mm: float
    solver: Mapping[str, Any]
    device: str

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> HydroShearConfig:
        if not isinstance(config, Mapping):
            raise TypeError("HydroShear config must be a mapping")
        allowed = {"gel_mesh", "surface_points", "output_grid", "solver", "device"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown HydroShear config keys: {sorted(unknown)}")
        if "gel_mesh" not in config:
            raise ValueError("HydroShear config requires a user-supplied closed gel_mesh")
        triangles = load_triangle_mesh(config["gel_mesh"])

        point_count = config.get("surface_points", 2048)
        if isinstance(point_count, bool) or not isinstance(point_count, (int, np.integer)):
            raise TypeError("surface_points must be an integer")
        if point_count <= 0:
            raise ValueError("surface_points must be positive")

        grid = config.get("output_grid", {})
        if not isinstance(grid, Mapping):
            raise TypeError("output_grid must be a mapping")
        grid_allowed = {
            "rows",
            "cols",
            "spacing_mm",
            "center_offset_mm",
            "marker_x_pixels",
            "marker_y_pixels",
            "pixel_scale_mm",
        }
        unknown_grid = set(grid) - grid_allowed
        if unknown_grid:
            raise ValueError(f"unknown output_grid keys: {sorted(unknown_grid)}")
        rows = _positive_int(grid.get("rows", 9), "output_grid.rows")
        cols = _positive_int(grid.get("cols", 7), "output_grid.cols")
        spacing = _pair(grid.get("spacing_mm", (2.145, 2.153125)), "spacing_mm")
        offset = _pair(grid.get("center_offset_mm", (-0.3575, -0.325)), "center_offset_mm")
        pixel_scale = _positive_float(grid.get("pixel_scale_mm", 0.065), "pixel_scale_mm")
        marker_x_pixels = grid.get("marker_x_pixels")
        marker_y_pixels = grid.get("marker_y_pixels")
        if (marker_x_pixels is None) != (marker_y_pixels is None):
            raise ValueError(
                "output_grid.marker_x_pixels and marker_y_pixels must be provided together"
            )
        if marker_x_pixels is None:
            x = (
                (np.arange(cols, dtype=np.float64) - (cols - 1) / 2) * spacing[0]
                + offset[0]
            )
            y = (
                (np.arange(rows, dtype=np.float64) - (rows - 1) / 2) * spacing[1]
                + offset[1]
            )
        else:
            x_pixels = _marker_pixels(marker_x_pixels, cols, CAMERA_W, "marker_x_pixels")
            y_pixels = _marker_pixels(marker_y_pixels, rows, CAMERA_H, "marker_y_pixels")
            x = (x_pixels - (CAMERA_W - 1) / 2) * pixel_scale
            y = (y_pixels - (CAMERA_H - 1) / 2) * pixel_scale
        pixel_x = x / pixel_scale + (CAMERA_W - 1) / 2
        pixel_y = y / pixel_scale + (CAMERA_H - 1) / 2
        if (
            np.any(pixel_x < 0)
            or np.any(pixel_x > CAMERA_W - 1)
            or np.any(pixel_y < 0)
            or np.any(pixel_y > CAMERA_H - 1)
        ):
            raise ValueError("output_grid coordinates lie outside the HydroShear camera view")

        raw_solver = config.get("solver", {})
        if not isinstance(raw_solver, Mapping):
            raise TypeError("solver must be a mapping")
        unknown_solver = set(raw_solver) - set(_SOLVER_DEFAULTS)
        if unknown_solver:
            raise ValueError(f"unknown HydroSoft solver parameters: {sorted(unknown_solver)}")
        solver = {**_SOLVER_DEFAULTS, **raw_solver}
        normal_axis = solver["normal_axis"]
        if (
            isinstance(normal_axis, bool)
            or not isinstance(normal_axis, (int, np.integer))
            or normal_axis != 2
        ):
            raise ValueError("HydroSoft currently supports only solver.normal_axis=2")
        solver["normal_axis"] = 2
        for name in (
            "mu",
            "lambda_d",
            "lambda_s",
            "shear_scale",
            "dilate_scale",
            "K",
            "E",
            "A",
        ):
            value = np.asarray(solver[name])
            if value.dtype.kind not in "iuf" or value.ndim not in (0, 1):
                raise TypeError(
                    f"solver.{name} must be a numeric scalar or one-dimensional environment array"
                )
            value = value.astype(np.float64, copy=False)
            if value.ndim == 1 and value.size == 0:
                raise ValueError(f"solver.{name} environment array must not be empty")
            if not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError(f"solver.{name} must contain finite non-negative values")
            solver[name] = float(value) if value.ndim == 0 else tuple(float(item) for item in value)
        for name in ("reverse_z", "use_fft_dilation", "use_matmul_shear"):
            if not isinstance(solver[name], bool):
                raise TypeError(f"solver.{name} must be bool")
        min_fft_points = solver["fft_dilation_min_pts"]
        if (
            isinstance(min_fft_points, bool)
            or not isinstance(min_fft_points, int)
            or min_fft_points <= 0
        ):
            raise ValueError("solver.fft_dilation_min_pts must be a positive integer")
        grid_shape = solver["fft_dilation_grid_shape"]
        if (
            not isinstance(grid_shape, Sequence)
            or isinstance(grid_shape, (str, bytes))
            or len(grid_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in grid_shape
            )
        ):
            raise ValueError("solver.fft_dilation_grid_shape must contain two positive integers")
        device = config.get("device", "cuda:0")
        if not isinstance(device, str) or not (
            device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit())
        ):
            raise ValueError("HydroShear requires a CUDA device such as 'cuda:0'")
        return cls(
            gel_triangles=triangles,
            surface_points=int(point_count),
            rows=rows,
            cols=cols,
            spacing_mm=spacing,
            center_offset_mm=offset,
            marker_x_mm=tuple(float(value) for value in x),
            marker_y_mm=tuple(float(value) for value in y),
            pixel_scale_mm=pixel_scale,
            solver=solver,
            device=device,
        )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a number, not bool")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _marker_pixels(value: Any, count: int, limit: int, name: str) -> np.ndarray:
    pixels = np.asarray(value)
    if pixels.dtype.kind not in "iuf" or pixels.ndim != 1 or pixels.size != count:
        raise ValueError(f"output_grid.{name} must contain {count} numeric pixel centers")
    pixels = pixels.astype(np.float64, copy=False)
    if (
        not np.isfinite(pixels).all()
        or np.any(pixels < 0.0)
        or np.any(pixels > limit - 1)
        or np.any(np.diff(pixels) <= 0.0)
    ):
        raise ValueError(
            f"output_grid.{name} must be finite, increasing, and inside the camera image"
        )
    return pixels


def _pair(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a finite [x, y] pair")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [x, y] pair")
    return float(result[0]), float(result[1])


def load_triangle_mesh(mesh: Any) -> np.ndarray:
    """Load closed gel triangles from an ``(N, 3, 3)`` array or Wavefront OBJ."""
    if isinstance(mesh, (str, Path)):
        triangles = _read_obj(Path(mesh).expanduser())
    else:
        triangles = np.asarray(mesh, dtype=np.float32)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not triangles.shape[0]:
        raise ValueError("gel_mesh must have shape (N, 3, 3), N > 0")
    triangles = np.ascontiguousarray(triangles, dtype=np.float32)
    if not np.isfinite(triangles).all():
        raise ValueError("gel_mesh must contain finite coordinates in metres")
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if np.any(areas <= 0.0):
        raise ValueError("gel_mesh triangles must have positive area")
    if not _is_closed(triangles):
        raise ValueError("gel_mesh must be a closed two-manifold triangle mesh")
    triangles.setflags(write=False)
    return triangles


def _read_obj(path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"OBJ vertex at line {line_number} needs three coordinates")
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
            elif fields[0] == "f":
                if len(fields) < 4:
                    raise ValueError(f"OBJ face at line {line_number} needs three vertices")
                indices = []
                for corner in fields[1:]:
                    raw = int(corner.split("/", 1)[0])
                    index = raw - 1 if raw > 0 else len(vertices) + raw
                    if index < 0 or index >= len(vertices):
                        raise ValueError(
                            f"OBJ face index outside vertex list at line {line_number}"
                        )
                    indices.append(index)
                faces.extend(
                    (indices[0], indices[index], indices[index + 1])
                    for index in range(1, len(indices) - 1)
                )
    if not vertices or not faces:
        raise ValueError(f"OBJ gel mesh {path} has no triangulatable faces")
    return np.asarray(vertices, dtype=np.float32)[np.asarray(faces, dtype=np.intp)]


def _is_closed(triangles: np.ndarray) -> bool:
    vertices = triangles.reshape(-1, 3)
    _, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(counts.size and np.all(counts == 2))
