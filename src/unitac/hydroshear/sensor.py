"""UniSim lifecycle and CUDA execution for the HydroShear model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from unisim.tactile import (
    TactileFieldSpec,
    TactileSceneDescription,
    TactileSensor,
    TactileSensorCapabilities,
)

from ..geometry import body_meshes, sample_object_surfaces
from .config import HydroShearConfig

_CAPABILITIES = TactileSensorCapabilities(
    pose_input_profiles=("numpy", "torch_cpu", "torch_cuda"),
    output_profiles=("torch_cuda",),
    supports_selected_environments=True,
    supports_geometry_rebuild=True,
    stream_event_ownership="caller",
)


def _assert_async(condition: torch.Tensor, message: str) -> None:
    assertion = getattr(torch, "_assert_async", None)
    if not callable(assertion):
        raise RuntimeError("Torch build does not provide asynchronous CUDA assertions")
    assertion(condition, message)


class _TorchStreamView:
    """Expose a Torch stream through the CUDA stream interoperability protocol."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __cuda_stream__(self) -> tuple[int, int]:
        return 0, int(self._stream.cuda_stream)


class HydroShearSensor(TactileSensor):
    """HydroShear tactile simulation over detached UniSim scene geometry.

    Inputs are world-space body positions in metres and unit ``wxyz`` poses.
    ``advance`` returns marker displacement in sensor-local coordinates in mm.
    CUDA is required for the SDF and camera depth stages; CPU poses are staged
    to that device while CUDA Torch poses stay on the caller's current stream.
    """

    sensor_type = "hydroshear"
    _tactile_capabilities = _CAPABILITIES

    def __init__(self, num_envs: int, config: Mapping[str, Any]) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        self._num_envs = num_envs
        self._config = HydroShearConfig.from_mapping(config)
        self._torch_device = torch.device(self._config.device)
        if not torch.cuda.is_available():
            raise RuntimeError("HydroShear SDF and depth computation require a working CUDA device")
        if self._torch_device.index is None:
            self._torch_device = torch.device("cuda", torch.cuda.current_device())
        if self._torch_device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {self._torch_device.index} is unavailable")
        torch.cuda.set_device(self._torch_device)
        try:
            import cupy as cp
        except ImportError as error:
            raise ImportError(
                "HydroShear requires cupy-cuda12x; install the 'hydroshear' extra"
            ) from error
        try:
            if cp.cuda.runtime.getDeviceCount() <= self._torch_device.index:
                raise RuntimeError("CuPy cannot access the configured CUDA device")
        except cp.cuda.runtime.CUDARuntimeError as error:
            raise RuntimeError("HydroShear requires a working CUDA runtime") from error
        self._cp = cp
        self._device_id = int(self._torch_device.index)
        self._scene: TactileSceneDescription | None = None
        self._body_pos: Any = None
        self._body_quat: Any = None
        self._object_meshes: tuple[np.ndarray, ...] = ()
        self._object_points: tuple[np.ndarray, ...] = ()
        self._object_triangles_gpu: tuple[Any, ...] = ()
        self._object_points_gpu: tuple[Any, ...] = ()
        self._pad_body_ids: tuple[int, ...] = ()
        self._object_body_ids: tuple[int, ...] = ()
        self._pad_frame_pos: Any = None
        self._pad_frame_quat: Any = None
        self._pad_pos: Any = None
        self._pad_quat: Any = None
        self._pad_inverse: Any = None
        self._object_pad: Any = None
        self._object_points_torch: Any = None
        self._marker_cp: Any = None
        self._marker_torch: Any = None
        self._height_cp: Any = None
        self._height_torch: Any = None
        self._depth_cp: Any = None
        self._depth_work: Any = None
        self._depth_renderer: Any = None
        self._marker_rows_cp: Any = None
        self._marker_cols_cp: Any = None
        self._marker_y_cp: Any = None
        self._marker_x_cp: Any = None
        self._output: Any = None
        self._solver: Any = None
        self._gel_sdf: Any = None
        self._timing_enabled = False
        self._stage_events: dict[str, tuple[Any, Any]] = {}
        self._transfer_stats = {
            "host_to_device_updates": 0,
            "host_to_device_bytes": 0,
            "device_to_host_updates": 0,
            "device_to_host_bytes": 0,
        }
        self._materialized = False
        self._closed = False
        self._pose_ready = False
        self._has_observation = False
        self._output_specs: tuple[TactileFieldSpec, ...] = ()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def output_specs(self) -> tuple[TactileFieldSpec, ...]:
        if not self._materialized:
            return ()
        return self._output_specs

    def get_tactile_capabilities(self) -> TactileSensorCapabilities:
        return _CAPABILITIES

    def materialize(self, scene: TactileSceneDescription) -> None:
        self._ensure_open()
        if self._materialized:
            raise RuntimeError("HydroShearSensor is already materialized; use rebuild(scene)")
        self._bind_scene(scene)
        self._pose_ready = True

    def rebuild(self, scene: TactileSceneDescription) -> None:
        self._ensure_open()
        if not isinstance(scene, TactileSceneDescription):
            raise TypeError("scene must be a TactileSceneDescription")
        old_pad_count = len(self._pad_body_ids)
        if old_pad_count and len(scene.sensor_frames) != old_pad_count:
            raise ValueError("geometry rebuild cannot change the number of mounted tactile frames")
        self._release_materialized()
        self._bind_scene(scene)

    def _bind_scene(self, scene: TactileSceneDescription) -> None:
        if not isinstance(scene, TactileSceneDescription):
            raise TypeError("scene must be a TactileSceneDescription")
        if not scene.object_body_names:
            raise ValueError("HydroShear requires at least one perceived object body")
        body_ids = {name: index for index, name in enumerate(scene.collision.scene.body_names)}
        missing = [
            frame.body_name for frame in scene.sensor_frames if frame.body_name not in body_ids
        ]
        if missing:
            raise ValueError(f"sensor mount bodies are absent from the scene: {missing}")
        self._scene = scene
        self._object_body_ids = tuple(body_ids[name] for name in scene.object_body_names)
        self._pad_body_ids = tuple(body_ids[frame.body_name] for frame in scene.sensor_frames)
        self._object_meshes = body_meshes(scene.collision, scene.object_body_names)
        self._object_points = sample_object_surfaces(
            self._object_meshes, self._config.surface_points
        )
        cp = self._cp
        flat_batch = self.num_envs * len(self._pad_body_ids)
        self._body_pos = cp.zeros(
            (self.num_envs, scene.collision.scene.num_bodies, 3), dtype=cp.float32
        )
        self._body_quat = cp.zeros(
            (self.num_envs, scene.collision.scene.num_bodies, 4), dtype=cp.float32
        )
        self._body_quat[..., 0] = 1.0
        self._pad_frame_pos = cp.asarray(
            np.stack([frame.pos for frame in scene.sensor_frames]), dtype=cp.float32
        )
        self._pad_frame_quat = cp.asarray(
            np.stack([frame.quat for frame in scene.sensor_frames]), dtype=cp.float32
        )
        self._pad_pos = cp.empty((flat_batch, 3), dtype=cp.float32)
        self._pad_quat = cp.empty((flat_batch, 4), dtype=cp.float32)
        self._pad_inverse = cp.empty_like(self._pad_quat)
        self._object_triangles_gpu = tuple(cp.asarray(mesh) for mesh in self._object_meshes)
        self._object_points_gpu = tuple(cp.asarray(points) for points in self._object_points)
        total_points = sum(points.shape[0] for points in self._object_points)
        self._object_pad = cp.empty((flat_batch, total_points, 3), dtype=cp.float32)
        self._object_points_torch = torch.from_dlpack(self._object_pad)
        self._depth_cp = cp.empty((flat_batch, 175, 100), dtype=cp.float32)
        self._depth_work = cp.empty_like(self._depth_cp) if len(self._object_body_ids) > 1 else None
        from .depth import DepthRenderer

        self._depth_renderer = DepthRenderer(flat_batch)

        x_mm = (
            np.arange(self._config.cols, dtype=np.float32) - (self._config.cols - 1) / 2
        ) * self._config.spacing_mm[0] + self._config.center_offset_mm[0]
        y_mm = (
            np.arange(self._config.rows, dtype=np.float32) - (self._config.rows - 1) / 2
        ) * self._config.spacing_mm[1] + self._config.center_offset_mm[1]
        yy, xx = np.meshgrid(y_mm / 1000.0, x_mm / 1000.0, indexing="ij")
        markers = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(-1, 3)
        self._marker_cp = cp.broadcast_to(
            cp.asarray(markers, dtype=cp.float32)[None, :, :],
            (flat_batch, markers.shape[0], 3),
        ).copy()
        rows = np.rint((y_mm / self._config.pixel_scale_mm + (320 - 1) / 2) * 175 / 320)
        cols = np.rint((x_mm / self._config.pixel_scale_mm + (240 - 1) / 2) * 100 / 240)
        self._marker_rows_cp = cp.asarray(np.clip(rows, 0, 174).astype(np.int32))
        self._marker_cols_cp = cp.asarray(np.clip(cols, 0, 99).astype(np.int32))
        self._marker_y_cp, self._marker_x_cp = cp.meshgrid(
            self._marker_rows_cp, self._marker_cols_cp, indexing="ij"
        )
        marker_count = self._config.rows * self._config.cols
        self._height_cp = cp.empty((flat_batch, marker_count), dtype=cp.float32)
        self._marker_torch = torch.from_dlpack(self._marker_cp)
        self._height_torch = torch.from_dlpack(self._height_cp)
        self._solver = self._create_solver(flat_batch, total_points)
        from .sdf import MeshSDFSensor

        self._gel_sdf = MeshSDFSensor(self._config.gel_triangles)
        self._gel_sdf.prepare(flat_batch * total_points)
        self._output = torch.zeros(
            (
                self.num_envs,
                len(self._pad_body_ids),
                self._config.rows,
                self._config.cols,
                3,
            ),
            dtype=torch.float32,
            device=self._torch_device,
        )
        self._output_specs = (
            TactileFieldSpec(
                name="displacement",
                shape=tuple(self._output.shape),
                dtype="float32",
                units="mm",
                frame="sensor_local",
                dimensions=("environment", "sensor", "row", "column", "component_xyz"),
                buffer_owner="sensor",
                buffer_lifetime="until_next_advance_reset_rebuild_or_close",
            ),
        )
        if self._timing_enabled:
            self._make_stage_events()
        self._materialized = True
        self._pose_ready = False
        self._has_observation = False

    def _create_solver(self, flat_batch: int, point_count: int):
        from .solver import HydroSoftSensor

        solver = HydroSoftSensor(device=str(self._torch_device), **self._config.solver)
        solver.initialize(flat_batch, point_count)
        return solver

    def update_pose(
        self,
        body_pos: Any,
        body_quat: Any,
        env_ids: Sequence[int] | Any | None = None,
    ) -> None:
        self._require_materialized()
        ids = _environment_rows(env_ids, self.num_envs)
        scene = self._scene
        if scene is None:
            raise RuntimeError("HydroShear scene is not materialized")
        expected_body_count = scene.collision.scene.num_bodies
        position, quaternion, profile = self._pose_inputs(body_pos, body_quat)
        if position.shape != (len(ids), expected_body_count, 3):
            raise ValueError(
                "body_pos must have shape "
                f"({len(ids)}, {expected_body_count}, 3), got {tuple(position.shape)}"
            )
        if quaternion.shape != (len(ids), expected_body_count, 4):
            raise ValueError(
                "body_quat must have shape "
                f"({len(ids)}, {expected_body_count}, 4), got {tuple(quaternion.shape)}"
            )
        cp = self._cp
        self._record_stage("pose_input", start=True)
        with self._caller_stream():
            if profile == "torch_cuda":
                pos_gpu = cp.from_dlpack(position.contiguous())
                quat_gpu = cp.from_dlpack(quaternion.contiguous())
                _assert_async(torch.isfinite(position).all(), "body_pos must be finite")
                _assert_async(torch.isfinite(quaternion).all(), "body_quat must be finite")
                quat_norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
                _assert_async((quat_norm >= 1e-12).all(), "body_quat must be non-zero")
                quat_gpu = quat_gpu / cp.maximum(
                    cp.linalg.norm(quat_gpu, axis=-1, keepdims=True), 1e-12
                )
            else:
                pos_host = np.asarray(position, dtype=np.float32)
                quat_host = np.asarray(quaternion, dtype=np.float32)
                if not np.isfinite(pos_host).all() or not np.isfinite(quat_host).all():
                    raise ValueError("body poses must be finite")
                quat_norm_host = np.linalg.norm(quat_host, axis=-1, keepdims=True)
                if np.any(quat_norm_host < 1e-12):
                    raise ValueError("body quaternions must be non-zero")
                pos_gpu = cp.asarray(pos_host, dtype=cp.float32)
                quat_gpu = cp.asarray(quat_host / quat_norm_host, dtype=cp.float32)
                self._transfer_stats["host_to_device_updates"] += 2
                self._transfer_stats["host_to_device_bytes"] += int(
                    pos_host.nbytes + quat_host.nbytes
                )
            self._body_pos[ids] = pos_gpu
            self._body_quat[ids] = quat_gpu
        self._record_stage("pose_input", start=False)
        self._pose_ready = True

    def _pose_inputs(self, body_pos: Any, body_quat: Any) -> tuple[Any, Any, str]:
        if isinstance(body_pos, np.ndarray) and isinstance(body_quat, np.ndarray):
            return body_pos, body_quat, "numpy"
        if isinstance(body_pos, torch.Tensor) and isinstance(body_quat, torch.Tensor):
            if body_pos.device != body_quat.device:
                raise TypeError("body_pos and body_quat must use the same device")
            if body_pos.device.type == "cpu":
                return body_pos.detach().numpy(), body_quat.detach().numpy(), "torch_cpu"
            if body_pos.device.type == "cuda":
                if body_pos.device.index != self._device_id:
                    raise ValueError(
                        f"pose tensors use cuda:{body_pos.device.index}; "
                        f"expected cuda:{self._device_id}"
                    )
                if body_pos.dtype != torch.float32 or body_quat.dtype != torch.float32:
                    raise TypeError("CUDA pose tensors must use float32")
                return body_pos, body_quat, "torch_cuda"
        raise TypeError("poses must be matching NumPy arrays or Torch tensors")

    def advance(self) -> None:
        self._require_materialized()
        if not self._pose_ready:
            raise RuntimeError("update_pose must be called after materialize or rebuild")
        cp = self._cp
        with cp.cuda.Device(self._device_id), self._caller_stream(), torch.no_grad():
            pad_count = len(self._pad_body_ids)
            flat_batch = self.num_envs * pad_count
            self._record_stage("transform", start=True)
            for pad_index, body_id in enumerate(self._pad_body_ids):
                body_quat = self._body_quat[:, body_id]
                site_pos = self._pad_frame_pos[pad_index]
                site_quat = self._pad_frame_quat[pad_index]
                pad_pos = self._body_pos[:, body_id] + _cp_quat_rotate(
                    body_quat, cp.broadcast_to(site_pos, (self.num_envs, 3))
                )
                pad_quat = _cp_quat_multiply(
                    body_quat, cp.broadcast_to(site_quat, (self.num_envs, 4))
                )
                self._pad_pos.reshape(self.num_envs, pad_count, 3)[:, pad_index] = pad_pos
                self._pad_quat.reshape(self.num_envs, pad_count, 4)[:, pad_index] = pad_quat
            self._pad_inverse[...] = _cp_quat_conjugate(self._pad_quat)

            transforms: list[tuple[Any, Any]] = []
            point_offset = 0
            for object_index, (body_id, local_points) in enumerate(
                zip(self._object_body_ids, self._object_points_gpu, strict=True)
            ):
                object_pos = cp.broadcast_to(
                    self._body_pos[:, body_id, None, :], (self.num_envs, pad_count, 3)
                ).reshape(flat_batch, 3)
                object_quat = cp.broadcast_to(
                    self._body_quat[:, body_id, None, :], (self.num_envs, pad_count, 4)
                ).reshape(flat_batch, 4)
                pad_pos = self._pad_pos
                relative_pos = _cp_quat_rotate(self._pad_inverse, object_pos - pad_pos)
                relative_quat = _cp_quat_multiply(self._pad_inverse, object_quat)
                rotations = _cp_quat_matrix(relative_quat)
                transforms.append((rotations, relative_pos))
                point_end = point_offset + local_points.shape[0]
                self._object_pad[:, point_offset:point_end] = (
                    _cp_quat_rotate(relative_quat[:, None, :], local_points[None, :, :])
                    + relative_pos[:, None, :]
                )
                point_offset = point_end

            self._record_stage("transform", start=False)
            self._record_stage("depth", start=True)
            for object_index, (triangles, (rotations, relative_pos)) in enumerate(
                zip(self._object_triangles_gpu, transforms, strict=True)
            ):
                target = self._depth_cp if object_index == 0 else self._depth_work
                if target is None:
                    raise RuntimeError("multi-object depth workspace was not materialized")
                self._depth_renderer.render_into(triangles, rotations, relative_pos, target)
                if object_index:
                    cp.minimum(self._depth_cp, self._depth_work, out=self._depth_cp)
            self._record_stage("depth", start=False)

            self._height_cp[...] = (
                -self._depth_cp[:, self._marker_y_cp, self._marker_x_cp].reshape(flat_batch, -1)
                / 1000.0
            )
            self._record_stage("sdf", start=True)
            sdf = self._gel_sdf.get_distance(self._object_points_torch)
            self._record_stage("sdf", start=False)
            self._record_stage("hydrosoft", start=True)
            _, displacement = self._solver.step_marker_displacement(
                self._marker_torch,
                self._height_torch,
                self._object_points_torch,
                sdf,
                reverse_z=self._config.solver["reverse_z"],
            )
            self._output.copy_(displacement.reshape_as(self._output))
            self._record_stage("hydrosoft", start=False)
        self._has_observation = True

    def enable_stage_timing(self, enabled: bool = True) -> None:
        """Enable optional CUDA-event timings used by the benchmark scripts."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        self._ensure_open()
        self._timing_enabled = enabled
        if enabled and self._materialized:
            self._make_stage_events()
        elif not enabled:
            self._stage_events.clear()

    @property
    def stage_timings_ms(self) -> Mapping[str, float]:
        """Return timings for the latest stages; reading synchronizes those events."""
        if not self._stage_events:
            return {}
        result = {}
        for name, (_, end) in self._stage_events.items():
            end.synchronize()
            start, _ = self._stage_events[name]
            result[name] = float(start.elapsed_time(end))
        return result

    @property
    def transfer_diagnostics(self) -> Mapping[str, int]:
        """Return pose staging counters accumulated by this sensor."""
        return dict(self._transfer_stats)

    def _make_stage_events(self) -> None:
        names = ("pose_input", "transform", "depth", "sdf", "hydrosoft")
        self._stage_events = {
            name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for name in names
        }

    def _record_stage(self, name: str, *, start: bool) -> None:
        if not self._timing_enabled:
            return
        begin, end = self._stage_events[name]
        (begin if start else end).record(torch.cuda.current_stream(self._torch_device))

    def read(self) -> Mapping[str, Any]:
        self._require_materialized()
        if not self._has_observation:
            raise RuntimeError("advance must complete before read")
        return {"displacement": self._output}

    def reset(self, env_ids: Sequence[int] | Any | None = None) -> None:
        self._require_materialized()
        rows = _environment_rows(env_ids, self.num_envs)
        if rows.size == 0:
            return
        pad_rows = (
            rows[:, None] * len(self._pad_body_ids)
            + np.arange(len(self._pad_body_ids), dtype=np.int64)[None, :]
        ).reshape(-1)
        row_tensor = torch.as_tensor(rows, dtype=torch.long, device=self._torch_device)
        pad_tensor = torch.as_tensor(pad_rows, dtype=torch.long, device=self._torch_device)
        with self._caller_stream():
            self._solver.reset(pad_tensor)
            self._depth_cp[pad_rows] = 0.0
            self._output[row_tensor] = 0.0
        self._has_observation = True

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self._torch_device)
        self._release_materialized()
        self._closed = True

    def _release_materialized(self) -> None:
        self._scene = None
        self._body_pos = None
        self._body_quat = None
        self._object_meshes = ()
        self._object_points = ()
        self._object_triangles_gpu = ()
        self._object_points_gpu = ()
        self._pad_body_ids = ()
        self._object_body_ids = ()
        for name in (
            "_pad_frame_pos",
            "_pad_frame_quat",
            "_pad_pos",
            "_pad_quat",
            "_pad_inverse",
            "_object_pad",
            "_object_points_torch",
            "_marker_cp",
            "_marker_torch",
            "_height_cp",
            "_height_torch",
            "_depth_cp",
            "_depth_work",
            "_depth_renderer",
            "_marker_rows_cp",
            "_marker_cols_cp",
            "_marker_y_cp",
            "_marker_x_cp",
            "_output",
            "_solver",
            "_gel_sdf",
        ):
            setattr(self, name, None)
        self._materialized = False
        self._pose_ready = False
        self._has_observation = False
        self._output_specs = ()
        self._stage_events.clear()

    def _caller_stream(self):
        current = torch.cuda.current_stream(self._torch_device)
        return self._cp.cuda.Stream.from_external(_TorchStreamView(current))

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("HydroShearSensor is closed")

    def _require_materialized(self) -> None:
        self._ensure_open()
        if not self._materialized:
            raise RuntimeError("HydroShearSensor must be materialized before this operation")


def _environment_rows(env_ids: Any, num_envs: int) -> np.ndarray:
    if env_ids is None:
        return np.arange(num_envs, dtype=np.int64)
    if isinstance(env_ids, torch.Tensor):
        if env_ids.device.type != "cpu":
            env_ids = env_ids.detach().cpu().numpy()
        else:
            env_ids = env_ids.detach().numpy()
    values = np.asarray(env_ids)
    if values.ndim != 1:
        raise ValueError("env_ids must be one-dimensional")
    if values.dtype == np.bool_:
        if values.shape != (num_envs,):
            raise ValueError(f"boolean env_ids must have length {num_envs}")
        return np.flatnonzero(values).astype(np.int64)
    if values.size and values.dtype.kind not in "iu":
        raise TypeError("env_ids must contain integers or a full-width boolean mask")
    rows = values.astype(np.int64, copy=False)
    if np.any(rows < 0) or np.any(rows >= num_envs) or np.unique(rows).size != rows.size:
        raise IndexError("env_ids must be distinct and within the environment batch")
    return rows


def _cp_quat_conjugate(quaternion: Any) -> Any:
    cp = __import__("cupy")
    result = cp.array(quaternion, dtype=cp.float32, copy=True)
    result[..., 1:] *= -1.0
    return result


def _cp_quat_multiply(left: Any, right: Any) -> Any:
    cp = __import__("cupy")
    lw, lx, ly, lz = cp.moveaxis(left, -1, 0)
    rw, rx, ry, rz = cp.moveaxis(right, -1, 0)
    return cp.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _cp_quat_rotate(quaternion: Any, vector: Any) -> Any:
    cp = __import__("cupy")
    uv = cp.cross(quaternion[..., 1:], vector)
    uuv = cp.cross(quaternion[..., 1:], uv)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


def _cp_quat_matrix(quaternion: Any) -> Any:
    cp = __import__("cupy")
    w, x, y, z = cp.moveaxis(quaternion, -1, 0)
    return cp.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((*quaternion.shape[:-1], 3, 3))
