"""MuJoCo playback rendering for the MJBatch and MJWarp examples."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .display import TactileDisplay
    from .simulation import CYCLE_FRAMES, LetterPressSimulation, observation_metrics, phase_target
else:
    from display import TactileDisplay
    from simulation import CYCLE_FRAMES, LetterPressSimulation, observation_metrics, phase_target


CAMERA_DISTANCE = 0.11
CAMERA_ELEVATION = -25.0
CAMERA_AZIMUTH = 45.0
CAMERA_LOOKAT = (0.0, 0.0, 0.008)


def _configure_free_camera(camera: Any, mujoco: Any) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = CAMERA_DISTANCE
    camera.elevation = CAMERA_ELEVATION
    camera.azimuth = CAMERA_AZIMUTH
    camera.lookat[:] = CAMERA_LOOKAT


class PlaybackRenderer:
    """Render detached UniSim physics snapshots with MuJoCo's visual model."""

    def __init__(
        self,
        simulation: LetterPressSimulation,
        *,
        width: int = 960,
        height: int = 720,
        offscreen: bool = True,
    ):
        import mujoco

        capabilities = simulation.backend.get_play_capabilities()
        if not capabilities.supports_physics_state_playback:
            raise RuntimeError(
                f"backend {simulation.backend_name!r} cannot render: UniSim does not "
                "declare physics-state playback; use --no-render"
            )
        source = simulation.backend.get_playback_model(env_index=0)
        if isinstance(source, mujoco.MjModel):
            self.model = copy.deepcopy(source)
        elif isinstance(source, (str, Path)):
            self.model = mujoco.MjModel.from_xml_path(str(source))
        else:
            raise RuntimeError(
                f"backend {simulation.backend_name!r} returned an unsupported playback "
                f"model type: {type(source).__name__}"
            )
        self.mujoco = mujoco
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        _configure_free_camera(self.camera, mujoco)
        self.renderer = (
            mujoco.Renderer(self.model, height=height, width=width) if offscreen else None
        )

    def sync(self, simulation: LetterPressSimulation) -> tuple[Any, float]:
        parts, _layout = simulation.playback_snapshot()
        if parts.qpos.shape[0] != 1:
            raise RuntimeError("letter playback expects exactly one environment")
        self.data.qpos[:] = parts.qpos[0]
        self.data.qvel[:] = parts.qvel[0]
        self.data.time = float(parts.time[0])
        if parts.mocap_pos is not None and parts.mocap_quat is not None:
            self.data.mocap_pos[:] = parts.mocap_pos[0]
            self.data.mocap_quat[:] = parts.mocap_quat[0]
        self.mujoco.mj_forward(self.model, self.data)
        return parts, float(parts.time[0])

    def render(self, simulation: LetterPressSimulation) -> tuple[np.ndarray, Any, float]:
        if self.renderer is None:
            raise RuntimeError("offscreen renderer was not created")
        parts, sim_time = self.sync(simulation)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy(), parts, sim_time

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


def _combined_frame(
    scene_rgb: np.ndarray, tactile_rgb: np.ndarray, phase: str, sim_time: float
) -> np.ndarray:
    import cv2

    scene = cv2.resize(scene_rgb, (960, 720), interpolation=cv2.INTER_AREA)
    cv2.putText(
        scene,
        f"{phase} | simulation {sim_time:.2f} s",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate((scene, tactile_rgb), axis=1)


def _write_image(cv2: Any, path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write image {path}")


def _metric_record(
    simulation: LetterPressSimulation,
    phase: str,
    target: np.ndarray,
    stats: dict[str, Any],
    observed: dict[str, Any],
    *,
    sim_time: float,
    total_step_ms: float,
    visualization_ms: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> dict[str, Any]:
    return {
        "backend": simulation.backend_name,
        "phase": phase,
        "simulation_time_s": sim_time,
        "target": target.tolist(),
        "qpos": qpos.tolist(),
        "qvel": qvel.tolist(),
        **observed,
        "physics_ms": stats["physics_ms"],
        "pose_read_transfer_ms": stats["pose_read_transfer_ms"],
        "tactile_ms": stats["tactile_ms"],
        "visualization_ms": visualization_ms,
        "total_step_ms": total_step_ms,
        "backend_step_timing_ms": stats["backend_timing"],
        "transfers": simulation.transfer_diagnostics(),
    }


def _reset_record(
    simulation: LetterPressSimulation,
    metrics_file: Any,
    *,
    target: np.ndarray,
) -> None:
    started = time.perf_counter()
    state = simulation.reset()
    reset_ms = (time.perf_counter() - started) * 1000.0
    assert simulation.sensor is not None
    snapshot = simulation.sensor.read_visualization_snapshot(env_id=0)
    observed = observation_metrics(
        snapshot.displacement[0], snapshot.depth_mm[0], snapshot.generation
    )
    record = {
        "backend": simulation.backend_name,
        "phase": "Reset",
        "simulation_time_s": 0.0,
        "target": target.tolist(),
        "qpos": state["qpos"].tolist(),
        "qvel": state["qvel"].tolist(),
        **observed,
        "reset_ms": reset_ms,
        "physics_ms": 0.0,
        "pose_read_transfer_ms": 0.0,
        "tactile_ms": 0.0,
        "visualization_ms": 0.0,
        "total_step_ms": reset_ms,
        "transfers": simulation.transfer_diagnostics(),
    }
    metrics_file.write(json.dumps(record, separators=(",", ":")) + "\n")


def run_headless(
    simulation: LetterPressSimulation,
    *,
    frames: int,
    output_dir: Path,
    mode: str,
    fps: float,
    calibration_dir: Path | None,
    display_gain: float,
) -> dict[str, Any]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "letter_press.mp4"
    metrics_path = output_dir / "metrics.jsonl"
    display = TactileDisplay(
        mode=mode,
        calibration_dir=calibration_dir,
        display_gain=display_gain,
    )
    playback = PlaybackRenderer(simulation)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not writer.isOpened():
        playback.close()
        display.close()
        raise RuntimeError(f"MP4 encoder could not open {video_path}")

    press_capture: tuple[int, np.ndarray, np.ndarray] | None = None
    rotate_capture: tuple[tuple[float, float], np.ndarray, np.ndarray] | None = None
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            _reset_record(simulation, metrics_file, target=np.zeros(2, dtype=np.float32))
            for frame_index in range(frames):
                if frame_index and frame_index % CYCLE_FRAMES == 0:
                    _reset_record(simulation, metrics_file, target=np.zeros(2, dtype=np.float32))
                phase, target = phase_target(frame_index)
                total_started = time.perf_counter()
                stats = simulation.advance_frame(target, capture_state=False)
                visualization_started = time.perf_counter()
                scene, parts, sim_time = playback.render(simulation)
                tactile = display.update(
                    simulation,
                    phase=phase,
                    sim_time=sim_time,
                )
                observed = tactile.metrics
                combined = _combined_frame(scene, tactile.panel_rgb, phase, sim_time)
                visualization_ms = (time.perf_counter() - visualization_started) * 1000.0
                writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                total_step_ms = (time.perf_counter() - total_started) * 1000.0
                qpos = parts.qpos[0].copy()
                qvel = parts.qvel[0].copy()
                record = _metric_record(
                    simulation,
                    phase,
                    target,
                    stats,
                    observed,
                    sim_time=sim_time,
                    total_step_ms=total_step_ms,
                    visualization_ms=visualization_ms,
                    qpos=qpos,
                    qvel=qvel,
                )
                metrics_file.write(json.dumps(record, separators=(",", ":")) + "\n")

                if phase == "Press":
                    score = int(observed["contact_pixels"])
                    if press_capture is None or score > press_capture[0]:
                        press_capture = (score, scene.copy(), tactile.image_rgb.copy())
                elif phase == "Rotate":
                    score = (
                        float(observed["peak_tangential_displacement_model_units"]),
                        float(qpos[1]),
                    )
                    if rotate_capture is None or score > rotate_capture[0]:
                        rotate_capture = (score, scene.copy(), tactile.image_rgb.copy())

            _reset_record(simulation, metrics_file, target=np.zeros(2, dtype=np.float32))
        if press_capture is not None:
            _write_image(cv2, output_dir / "scene_press.png", press_capture[1])
            _write_image(cv2, output_dir / "tactile_press.png", press_capture[2])
        if rotate_capture is not None:
            _write_image(cv2, output_dir / "scene_rotate.png", rotate_capture[1])
            _write_image(cv2, output_dir / "tactile_rotate.png", rotate_capture[2])
    finally:
        writer.release()
        playback.close()
        display.close()

    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"MP4 recording failed: {video_path}")
    return {
        "backend": simulation.backend_name,
        "frames": frames,
        "video": str(video_path),
        "metrics": str(metrics_path),
        "screenshots": [
            str(path)
            for path in (
                output_dir / "scene_press.png",
                output_dir / "tactile_press.png",
                output_dir / "scene_rotate.png",
                output_dir / "tactile_rotate.png",
            )
            if path.is_file()
        ],
    }


def run_interactive(
    simulation: LetterPressSimulation,
    *,
    frames: int | None,
    fps: float,
    mode: str,
    calibration_dir: Path | None,
    display_gain: float,
) -> None:
    import mujoco
    import mujoco.viewer

    display = TactileDisplay(
        mode=mode,
        calibration_dir=calibration_dir,
        display_gain=display_gain,
    )
    playback = PlaybackRenderer(simulation, offscreen=False)
    state = {"paused": False, "restart": False, "single_step": False}

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (ord("r"), ord("R")):
            state["restart"] = True
        elif keycode in (ord("n"), ord("N"), ord(".")):
            state["single_step"] = True
        elif keycode == ord("1"):
            display.set_mode("rgb")
        elif keycode == ord("2"):
            display.set_mode("depth")

    viewer = None
    try:
        viewer = mujoco.viewer.launch_passive(
            playback.model,
            playback.data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
        with viewer.lock():
            _configure_free_camera(viewer.cam, mujoco)

        frame_index = 0
        last_cycle = 0
        while viewer.is_running() and (frames is None or frame_index < frames):
            started = time.perf_counter()
            single_step = bool(state["single_step"])
            if state["restart"]:
                simulation.reset()
                frame_index = 0
                last_cycle = 0
                state["restart"] = False
            if not state["paused"] or single_step:
                cycle = frame_index // CYCLE_FRAMES
                if cycle > last_cycle:
                    simulation.reset()
                    last_cycle = cycle
                phase, target = phase_target(frame_index)
                simulation.advance_frame(
                    target,
                    substeps=1 if single_step else 10,
                    capture_state=False,
                )
                if not single_step:
                    frame_index += 1
            else:
                phase, _target = phase_target(max(frame_index - 1, 0))
            state["single_step"] = False

            with viewer.lock():
                _parts, sim_time = playback.sync(simulation)
            tactile = display.update(simulation, phase=phase, sim_time=sim_time)
            viewport = viewer.viewport
            if viewport is not None:
                import cv2

                scale = min(
                    1.0,
                    max(1, viewport.width - 10) / 320,
                    max(1, viewport.height - 10) / 720,
                )
                width = max(1, int(round(320 * scale)))
                height = max(1, int(round(720 * scale)))
                panel = cv2.resize(tactile.panel_rgb, (width, height), interpolation=cv2.INTER_AREA)
                viewer.set_images(
                    (
                        mujoco.MjrRect(
                            viewport.left + viewport.width - width - 10,
                            viewport.bottom + (viewport.height - height) // 2,
                            width,
                            height,
                        ),
                        panel,
                    )
                )
            font = mujoco.mjtFontScale.mjFONTSCALE_150
            viewer.set_texts(
                [
                    (
                        font,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f"Backend: {simulation.backend_name}",
                        "",
                    ),
                ]
            )
            viewer.sync()
            elapsed = time.perf_counter() - started
            delay = 1.0 / fps - elapsed
            if delay > 0:
                time.sleep(delay)
    finally:
        if viewer is not None and viewer.is_running():
            viewer.close()
        while viewer is not None and viewer.m is not None:
            time.sleep(0.01)
        playback.close()
        display.close()
