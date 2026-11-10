"""UniSim physics and HydroShear lifecycle for the letter press example."""

from __future__ import annotations

import tempfile
import time
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from unisim import TensorIOSpec
from unisim.backend.process_device import configure_backend_process_device
from unisim.factory import create_backend, create_tactile_sensor
from unisim.scene import SceneCfg
from unisim.tactile import TactileSceneDescription

from unitac.hydroshear.config import load_triangle_mesh

BODY_NAMES = ("pad", "indenter")
SITE_NAMES = ("gel_surface",)
ACTUATOR_NAMES = ("press_actuator", "rotate_actuator")
SIM_DT = 0.002
SUBSTEPS_PER_FRAME = 10
PHASES = (
    ("Approach", 20),
    ("Press", 40),
    ("Hold", 30),
    ("Rotate", 60),
    ("Lift", 40),
    ("Return", 40),
)
CYCLE_FRAMES = sum(duration for _, duration in PHASES)


@contextmanager
def scene_files() -> Iterator[Path]:
    """Materialize the packaged MJCF assets for path-based backend loaders."""
    asset_root = resources.files("unitac.hydroshear").joinpath("assets", "letter_press")
    with tempfile.TemporaryDirectory(prefix="unitac-letter-press-") as temporary:
        root = Path(temporary)
        for name in ("scene.xml", "letter.STL", "gsmini_elastomer_collision_512.obj"):
            (root / name).write_bytes(asset_root.joinpath(name).read_bytes())
        yield root / "scene.xml"


def phase_target(frame_index: int) -> tuple[str, np.ndarray]:
    """Return the existing smooth press and rotation actuator trajectory."""
    index = frame_index % CYCLE_FRAMES
    start = 0
    for phase, duration in PHASES:
        if index < start + duration:
            local = (index - start) / max(1, duration - 1)
            smooth = local * local * (3.0 - 2.0 * local)
            if phase == "Approach":
                press, rotate = 0.0, 0.0
            elif phase == "Press":
                press, rotate = -0.002 * smooth, 0.0
            elif phase in ("Hold", "Rotate"):
                press = -0.002
                rotate = 0.5235987756 * smooth if phase == "Rotate" else 0.0
            elif phase == "Lift":
                press, rotate = -0.002 + 0.005 * smooth, 0.5235987756
            else:
                press, rotate = 0.003 * (1.0 - smooth), 0.5235987756 * (1.0 - smooth)
            return phase, np.asarray((press, rotate), dtype=np.float32)
        start += duration
    raise RuntimeError("letter press phase schedule is inconsistent")


class LetterPressSimulation:
    """One physical backend feeding the backend-neutral HydroShear sensor."""

    def __init__(self, backend_name: str, xml_path: Path, device: str) -> None:
        self._hydroshear_device = torch.device(device)
        if self._hydroshear_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("HydroShear requires an available CUDA device")
        if (
            self._hydroshear_device.index is not None
            and self._hydroshear_device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError(
                f"CUDA device {self._hydroshear_device.index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)"
            )
        configure_backend_process_device(backend_name, device)
        if self._hydroshear_device.index is None:
            self._hydroshear_device = torch.device("cuda", torch.cuda.current_device())
        self.backend = create_backend(
            backend_name,
            SceneCfg(model_file=str(xml_path)),
            num_envs=1,
            sim_dt=SIM_DT,
            base_name="indenter",
            body_state_required=True,
        )
        self.sensor: Any | None = None
        self.pose_plan: Any | None = None
        self._closed = False
        self.physics_steps = 0
        with torch.cuda.device(self._hydroshear_device):
            self._tactile_start = torch.cuda.Event(enable_timing=True)
            self._tactile_end = torch.cuda.Event(enable_timing=True)
        try:
            self.backend.materialize()
            self.backend_name = str(self.backend.backend_type)
            self._initialize(xml_path, device)
        except BaseException:
            self.close()
            raise

    def _initialize(self, xml_path: Path, device: str) -> None:
        capabilities = self.backend.get_tensor_capabilities()
        if not capabilities.stepping or not capabilities.state_views:
            raise RuntimeError(
                f"UniSim backend {self.backend_name!r} lacks tensor stepping or state views"
            )

        actuator_names = tuple(self.backend.get_actuator_names())
        if actuator_names != ACTUATOR_NAMES:
            raise RuntimeError(
                f"backend {self.backend_name!r} actuator order is {actuator_names}; "
                f"expected {ACTUATOR_NAMES}"
            )

        collision = self.backend.get_ray_collision_description(body_names=BODY_NAMES)
        body_names = tuple(collision.scene.body_names)
        if body_names != BODY_NAMES:
            raise RuntimeError(
                f"backend {self.backend_name!r} collision body order is {body_names}; "
                f"expected {BODY_NAMES}"
            )
        try:
            geometry_state = self.backend.get_ray_geometry_state(body_names=BODY_NAMES)
        except NotImplementedError:
            # The packaged letter is a fixed mesh attached to a rigid body.
            # Its local geometry and transform are fully described by collision.
            geometry_state = None
        site_frames = self.backend.get_ray_site_frames(SITE_NAMES)
        if tuple(frame.site_name for frame in site_frames) != SITE_NAMES:
            raise RuntimeError(f"backend {self.backend_name!r} returned unexpected sites")
        if tuple(frame.body_name for frame in site_frames) != ("pad",):
            raise RuntimeError(
                f"backend {self.backend_name!r} attached gel_surface to the wrong body"
            )

        gel_triangles = load_triangle_mesh(
            xml_path.parent / "gsmini_elastomer_collision_512.obj"
        ).copy()
        gel_triangles += np.asarray((0.0005, 0.05975, 0.00725), dtype=np.float32)
        tactile_scene = TactileSceneDescription(
            collision=collision,
            object_body_names=("indenter",),
            sensor_frames=site_frames,
            object_geom_names=("letter_geom",),
            geometry_state=geometry_state,
        )
        self.sensor = create_tactile_sensor(
            "unitac",
            model="hydroshear",
            num_envs=1,
            config={
                "gel_mesh": gel_triangles,
                "surface_points": 2048,
                "output_grid": {
                    "rows": 9,
                    "cols": 7,
                    "spacing_mm": [2.145, 2.153125],
                    "center_offset_mm": [-0.3575, -0.325],
                    "marker_x_pixels": [15, 49, 81, 115, 146, 179, 213],
                    "marker_y_pixels": [22, 57, 91, 125, 158, 191, 223, 255, 287],
                    "pixel_scale_mm": 0.065,
                },
                "solver": {"use_fft_dilation": False, "use_matmul_shear": True},
                "device": device,
            },
        )
        self.sensor.materialize(tactile_scene)

        if capabilities.tracked_body_views:
            self.pose_source = "tracked_body_views"
        elif capabilities.packed_host_bridge:
            self.pose_source = "host_bridge"
            self.pose_plan = self.backend.bind_host_bridge_io(
                TensorIOSpec(
                    body_names=body_names,
                    body_fields=("pos_w", "quat_w"),
                    device="cpu",
                )
            )
        else:
            raise RuntimeError(
                f"backend {self.backend_name!r} cannot expose body poses through a "
                "supported UniSim tensor capability"
            )

        state = self.backend.get_state_views(("qpos", "qvel"))
        self._state_device = state["qpos"].device
        self._initial_qpos = state["qpos"].clone().contiguous()
        self._initial_qvel = state["qvel"].clone().contiguous()
        if tuple(self._initial_qpos.shape) != (1, 2) or tuple(self._initial_qvel.shape) != (1, 2):
            raise RuntimeError(
                f"backend {self.backend_name!r} exposed state shapes "
                f"{tuple(self._initial_qpos.shape)} and {tuple(self._initial_qvel.shape)}; "
                "the letter scene requires two scalar joints"
            )

    def _body_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pose_source == "tracked_body_views":
            views = self.backend.get_tracked_body_views(BODY_NAMES)
        else:
            assert self.pose_plan is not None
            self.pose_plan.read_state_sensors()
            views = self.pose_plan.get_body_pose_views()
        if tuple(views.body_names) != BODY_NAMES:
            raise RuntimeError(f"backend {self.backend_name!r} returned body poses out of order")
        if views.pos_w is None or views.quat_w is None:
            raise RuntimeError(f"backend {self.backend_name!r} returned incomplete body poses")
        return views.pos_w, views.quat_w

    def _wait_for_step(self, result: Any) -> dict[str, float]:
        if isinstance(result, dict):
            completion = result.get("completion")
            if completion is not None:
                completion.wait_before_read()
            timing = result.get("timing", {})
            if isinstance(timing, dict):
                return {
                    name: float(value)
                    for name, value in timing.items()
                    if isinstance(value, (int, float))
                }
        return {}

    def _advance_substep(self, target: np.ndarray) -> dict[str, float]:
        control = torch.as_tensor(
            target[None, :], dtype=torch.float32, device=self._state_device
        ).contiguous()
        physics_start = time.perf_counter()
        result = self.backend.step_tensor(control, nsteps=1)
        backend_timing = self._wait_for_step(result)
        physics_ms = (time.perf_counter() - physics_start) * 1000.0

        pose_start = time.perf_counter()
        position, quaternion = self._body_poses()
        assert self.sensor is not None
        self.sensor.update_pose(position, quaternion)
        pose_ms = (time.perf_counter() - pose_start) * 1000.0

        stream = torch.cuda.current_stream(self._hydroshear_device)
        self._tactile_start.record(stream)
        self.sensor.advance()
        self._tactile_end.record(stream)
        self._tactile_end.synchronize()
        tactile_ms = float(self._tactile_start.elapsed_time(self._tactile_end))
        self.physics_steps += 1
        return {
            "physics_ms": physics_ms,
            "pose_read_transfer_ms": pose_ms,
            "tactile_ms": tactile_ms,
            "backend_timing": backend_timing,
        }

    def advance_frame(
        self,
        target: np.ndarray,
        substeps: int = SUBSTEPS_PER_FRAME,
        *,
        capture_state: bool = True,
    ) -> dict[str, Any]:
        totals: dict[str, Any] = {
            "physics_ms": 0.0,
            "pose_read_transfer_ms": 0.0,
            "tactile_ms": 0.0,
            "backend_timing": {},
        }
        for _ in range(substeps):
            timing = self._advance_substep(target)
            for name in ("physics_ms", "pose_read_transfer_ms", "tactile_ms"):
                totals[name] += timing[name]
            for name, value in timing["backend_timing"].items():
                totals["backend_timing"][name] = (
                    totals["backend_timing"].get(name, 0.0) + value
                )

        if capture_state:
            state_start = time.perf_counter()
            state = self.backend.get_state_views(("qpos", "qvel"))
            totals["pose_read_transfer_ms"] += (time.perf_counter() - state_start) * 1000.0
            totals["qpos"] = state["qpos"].detach().cpu().numpy()[0].copy()
            totals["qvel"] = state["qvel"].detach().cpu().numpy()[0].copy()
        totals["simulation_time_s"] = self.physics_steps * SIM_DT
        totals["physics_steps"] = self.physics_steps
        totals["backend"] = self.backend_name
        return totals

    def reset(self) -> dict[str, Any]:
        env_ids = torch.zeros((1,), dtype=torch.int64, device=self._state_device)
        self.backend.set_state_tensor(
            env_ids,
            self._initial_qpos.clone().contiguous(),
            self._initial_qvel.clone().contiguous(),
        )
        assert self.sensor is not None
        self.sensor.reset()
        self.physics_steps = 0
        state = self.backend.get_state_views(("qpos", "qvel"))
        return {
            "backend": self.backend_name,
            "qpos": state["qpos"].detach().cpu().numpy()[0].copy(),
            "qvel": state["qvel"].detach().cpu().numpy()[0].copy(),
            "simulation_time_s": 0.0,
        }

    def playback_snapshot(self) -> tuple[Any, Any]:
        capabilities = self.backend.get_play_capabilities()
        if not capabilities.supports_physics_state_playback:
            raise RuntimeError(
                f"backend {self.backend_name!r} does not support physics-state playback"
            )
        snapshot = self.backend.get_physics_state()
        layout = self.backend.get_physics_state_layout()
        return layout.split_state(snapshot), layout

    def transfer_diagnostics(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "backend": dict(self.backend.get_tensor_transfer_diagnostics()),
            "sensor_pose": dict(self.sensor.transfer_diagnostics) if self.sensor else {},
        }
        if self.pose_plan is not None:
            values["host_bridge"] = dict(self.pose_plan.transfer_stats)
        return values

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.pose_plan is not None:
                self.pose_plan.close()
        finally:
            try:
                if self.sensor is not None:
                    self.sensor.close()
            finally:
                self.backend.close()

    def __enter__(self) -> LetterPressSimulation:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def observation_metrics(
    displacement: np.ndarray, depth_mm: np.ndarray, generation: int
) -> dict[str, Any]:
    """Summarize one copied tactile frame without changing model units."""
    raw = np.asarray(displacement, dtype=np.float32)
    depth = np.asarray(depth_mm, dtype=np.float32)
    mean = raw.mean(axis=(0, 1))
    tangent = np.linalg.norm(raw[..., :2], axis=-1)
    return {
        "contact_pixels": int(np.count_nonzero(depth < -0.05)),
        "peak_depth_mm": float(max(0.0, -float(depth.min()))),
        "peak_displacement_model_units": float(np.linalg.norm(raw, axis=-1).max()),
        "peak_tangential_displacement_model_units": float(tangent.max()),
        "mean_displacement_model_units": mean.tolist(),
        "mean_tangential_displacement_model_units": mean[:2].tolist(),
        "generation": int(generation),
        "finite": bool(np.isfinite(raw).all() and np.isfinite(depth).all()),
    }


def tensor_observation_metrics(displacement: torch.Tensor) -> dict[str, Any]:
    """Reduce the public CUDA observation to a few scalars for no-render logs."""
    raw = displacement[0].to(dtype=torch.float32)
    magnitude = torch.linalg.vector_norm(raw, dim=-1)
    tangent = torch.linalg.vector_norm(raw[..., :2], dim=-1)
    mean = raw.mean(dim=(0, 1, 2))
    summary = torch.stack(
        (
            torch.isfinite(raw).all().to(dtype=torch.float32),
            magnitude.max(),
            tangent.max(),
            mean[0],
            mean[1],
            mean[2],
        )
    ).detach().cpu().tolist()
    return {
        "contact_pixels": None,
        "peak_depth_mm": None,
        "peak_displacement_model_units": float(summary[1]),
        "peak_tangential_displacement_model_units": float(summary[2]),
        "mean_displacement_model_units": [float(value) for value in summary[3:]],
        "mean_tangential_displacement_model_units": [float(value) for value in summary[3:5]],
        "finite": bool(summary[0]),
    }
