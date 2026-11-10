"""Run the real UniSim/MuJoCo letter press with HydroShear visualization."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from unisim import TensorIOSpec
from unisim.factory import create_backend, create_tactile_sensor
from unisim.scene import SceneCfg
from unisim.tactile import TactileSceneDescription

from unitac.geometry import body_meshes
from unitac.hydroshear.config import load_triangle_mesh
from unitac.visualization import create_visualizer

BODY_NAMES = ("pad", "indenter")
SITE_NAMES = ("gel_surface",)
SUBSTEPS_PER_FRAME = 10
FRAME_RATE = 30.0
_PHASES = (
    ("Approach", 20),
    ("Press", 40),
    ("Hold", 30),
    ("Rotate", 60),
    ("Lift", 40),
    ("Return", 40),
)
_CYCLE_FRAMES = sum(duration for _, duration in _PHASES)


@contextmanager
def _scene_files() -> Iterator[Path]:
    """Materialize package resources for MuJoCo's path-based MJCF loader."""
    from importlib import resources

    resource_root = resources.files("unitac.hydroshear").joinpath("assets", "letter_press")
    with tempfile.TemporaryDirectory(prefix="unitac-letter-press-") as temporary:
        root = Path(temporary)
        for name in ("scene.xml", "letter.STL", "gsmini_elastomer_collision_512.obj"):
            (root / name).write_bytes(resource_root.joinpath(name).read_bytes())
        yield root / "scene.xml"


def _phase_target(frame_index: int) -> tuple[str, np.ndarray]:
    """Return smooth actuator targets for the scripted contact cycle."""
    index = frame_index % _CYCLE_FRAMES
    start = 0
    for phase, duration in _PHASES:
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
                press, rotate = -0.002 * (1.0 - smooth), 0.5235987756
            else:
                press, rotate = 0.0, 0.5235987756 * (1.0 - smooth)
            return phase, np.asarray([press, rotate], dtype=np.float32)
        start += duration
    raise AssertionError("phase schedule is inconsistent")


def _reset(
    backend: object,
    sensor: object,
    initial_qpos: torch.Tensor,
    initial_qvel: torch.Tensor,
) -> None:
    backend.set_state_tensor(
        torch.tensor([0], dtype=torch.int64), initial_qpos.contiguous(), initial_qvel.contiguous()
    )
    backend.set_control_targets(np.asarray([0], dtype=np.int64), np.zeros((1, 2), dtype=np.float64))
    sensor.reset()


def _build_simulation(xml_path: Path, device: str):
    backend = create_backend(
        "mujoco",
        SceneCfg(model_file=str(xml_path)),
        num_envs=1,
        sim_dt=0.002,
        base_name="indenter",
        tracked_body_names=BODY_NAMES,
        body_state_required=True,
        add_body_sensors=True,
    )
    sensor = None
    pose_plan = None
    try:
        backend.materialize()
        collision = backend.get_ray_collision_description(body_names=BODY_NAMES)
        geometry_state = backend.get_ray_geometry_state(body_names=BODY_NAMES)
        site_frames = backend.get_ray_site_frames(SITE_NAMES)
        if tuple(frame.site_name for frame in site_frames) != SITE_NAMES:
            raise RuntimeError(f"unexpected tactile sites: {site_frames}")
        triangles = load_triangle_mesh(
            xml_path.parent / "gsmini_elastomer_collision_512.obj"
        ).copy()
        triangles += np.asarray([0.0005, 0.05975, 0.00725], dtype=np.float32)
        hydro_config = {
            "gel_mesh": triangles,
            "surface_points": 2048,
            "output_grid": {
                "rows": 9,
                "cols": 7,
                "spacing_mm": [2.145, 2.153125],
                "center_offset_mm": [-0.3575, -0.325],
                "pixel_scale_mm": 0.065,
            },
            "solver": {"use_fft_dilation": False, "use_matmul_shear": True},
            "device": device,
        }
        tactile_scene = TactileSceneDescription(
            collision=collision,
            object_body_names=("indenter",),
            sensor_frames=site_frames,
            object_geom_names=("letter_geom",),
            geometry_state=geometry_state,
        )
        sensor = create_tactile_sensor(
            "unitac", model="hydroshear", num_envs=1, config=hydro_config
        )
        sensor.materialize(tactile_scene)
        pose_plan = backend.bind_host_bridge_io(
            TensorIOSpec(
                body_names=BODY_NAMES,
                body_fields=("pos_w", "quat_w"),
                device="cpu",
            )
        )
        collision_mesh = body_meshes(
            collision,
            ("indenter",),
            geom_names=("letter_geom",),
            geometry_state=geometry_state,
        )[0]
        if not np.isfinite(collision_mesh).all():
            raise RuntimeError("letter collision geometry contains non-finite coordinates")
        initial = backend.get_state_views(("qpos", "qvel"), device="cpu")
        return backend, sensor, pose_plan, initial["qpos"].clone(), initial["qvel"].clone()
    except Exception:
        if pose_plan is not None:
            pose_plan.close()
        if sensor is not None:
            sensor.close()
        backend.close()
        raise


def _one_physics_step(
    backend: object, sensor: object, pose_plan: object, target: np.ndarray
) -> None:
    backend.set_control_targets(np.asarray([0], dtype=np.int64), target[None, :].astype(np.float64))
    control = torch.as_tensor(target[None, :], dtype=torch.float32, device="cpu").contiguous()
    backend.step_tensor(control, nsteps=1)
    pose_plan.read_state_sensors()
    poses = pose_plan.get_body_pose_views()
    sensor.update_pose(poses.pos_w, poses.quat_w)
    sensor.advance()


def _sync_render_data(model: object, data: object, backend: object, simulation_steps: int) -> None:
    state = backend.get_state_views(("qpos", "qvel", "ctrl"), device="cpu")
    data.qpos[:] = state["qpos"][0].numpy()
    data.qvel[:] = state["qvel"][0].numpy()
    data.ctrl[:] = state["ctrl"][0].numpy()
    data.time = simulation_steps * float(model.opt.timestep)
    import mujoco

    mujoco.mj_forward(model, data)


def _tactile_panel(
    image_rgb: np.ndarray,
    mode: str,
    metrics: dict[str, float],
    calibration_label: str,
    marker_label: str,
) -> np.ndarray:
    import cv2

    panel = np.full((720, 320, 3), (25, 28, 33), dtype=np.uint8)
    cv2.putText(
        panel,
        f"{calibration_label} | {mode.upper()}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        marker_label,
        (12, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (80, 225, 255),
        1,
        cv2.LINE_AA,
    )
    image = cv2.resize(image_rgb, (300, 400), interpolation=cv2.INTER_AREA)
    panel[64:464, 10:310] = image
    lines = (
        f"Contact pixels: {metrics['contact_pixels']:.0f}",
        f"Peak depth: {metrics['peak_depth_mm']:.3f} mm",
        f"Peak marker: {metrics['peak_displacement_mm']:.3f} mm",
        f"Sensor generation: {metrics['generation']:.0f}",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (16, 500 + index * 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (225, 230, 235),
            1,
            cv2.LINE_AA,
        )
    return panel


def _combined_frame(
    scene_rgb: np.ndarray, tactile_rgb: np.ndarray, phase: str, sim_time: float
) -> np.ndarray:
    import cv2

    scene = cv2.resize(scene_rgb, (960, 720), interpolation=cv2.INTER_AREA)
    panel = tactile_rgb.copy()
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
    return np.concatenate((scene, panel), axis=1)


def run_headless(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("the letter press HydroShear example requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import cv2
    import mujoco

    visualizer = create_visualizer(
        "hydroshear",
        mode=args.mode,
        profile=args.profile,
        calibration_dir=args.calibration_dir,
        display_gain=args.display_gain,
    )
    output_video = args.output_dir / "letter_press.mp4"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), FRAME_RATE, (1280, 720)
    )
    if not writer.isOpened():
        visualizer.close()
        raise RuntimeError(f"MP4 encoder could not open {output_video}")

    metrics_path = args.output_dir / "metrics.jsonl"
    try:
        with _scene_files() as xml_path:
            backend, sensor, pose_plan, initial_qpos, initial_qvel = _build_simulation(
                xml_path, args.device
            )
            model = backend.model
            render_model = mujoco.MjModel.from_xml_path(str(xml_path))
            render_data = mujoco.MjData(render_model)
            renderer = mujoco.Renderer(render_model, height=720, width=960)
            try:
                _reset(backend, sensor, initial_qpos, initial_qvel)
                simulation_steps = 0
                saved_phases: set[str] = set()
                with metrics_path.open("w", encoding="utf-8") as metrics_file:
                    for frame_index in range(args.frames):
                        if frame_index and frame_index % _CYCLE_FRAMES == 0:
                            sensor.reset()
                        phase, target = _phase_target(frame_index)
                        for _ in range(SUBSTEPS_PER_FRAME):
                            _one_physics_step(backend, sensor, pose_plan, target)
                            simulation_steps += 1
                        snapshot_frame = visualizer.update(sensor, env_id=0)
                        contact_pixels = int(np.count_nonzero(snapshot_frame.depth_mm < -0.05))
                        sim_time = simulation_steps * float(model.opt.timestep)
                        qpos = backend.get_state_views(("qpos",), device="cpu")["qpos"][0].tolist()
                        mean_displacement = snapshot_frame.raw_displacement_mm.mean(axis=(0, 1))
                        metrics = {
                            "phase": phase,
                            "simulation_time_s": sim_time,
                            "contact_pixels": contact_pixels,
                            "peak_depth_mm": snapshot_frame.peak_depth_mm,
                            "peak_displacement_mm": snapshot_frame.peak_displacement_mm,
                            "mean_displacement_mm": mean_displacement.tolist(),
                            "mean_tangential_displacement_mm": mean_displacement[:2].tolist(),
                            "generation": snapshot_frame.generation,
                            "qpos": qpos,
                        }
                        metrics_file.write(json.dumps(metrics, separators=(",", ":")) + "\n")
                        _sync_render_data(render_model, render_data, backend, simulation_steps)
                        renderer.update_scene(render_data, camera="overview")
                        scene_image = renderer.render().copy()
                        panel = _tactile_panel(
                            snapshot_frame.image_rgb,
                            args.mode,
                            metrics,
                            snapshot_frame.calibration_label,
                            snapshot_frame.marker_label,
                        )
                        combined = _combined_frame(scene_image, panel, phase, sim_time)
                        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                        if frame_index == 0:
                            cv2.imwrite(
                                str(args.output_dir / "scene.png"),
                                cv2.cvtColor(scene_image, cv2.COLOR_RGB2BGR),
                            )
                            cv2.imwrite(
                                str(args.output_dir / f"tactile_{args.mode}.png"),
                                cv2.cvtColor(snapshot_frame.image_rgb, cv2.COLOR_RGB2BGR),
                            )
                        if phase not in saved_phases:
                            cv2.imwrite(
                                str(args.output_dir / f"scene_{phase.lower()}.png"),
                                cv2.cvtColor(scene_image, cv2.COLOR_RGB2BGR),
                            )
                            cv2.imwrite(
                                str(args.output_dir / f"tactile_{args.mode}_{phase.lower()}.png"),
                                cv2.cvtColor(snapshot_frame.image_rgb, cv2.COLOR_RGB2BGR),
                            )
                            saved_phases.add(phase)
                        save_contact = (
                            phase == "Press"
                            and contact_pixels > 0
                            and "press_contact" not in saved_phases
                        )
                        save_rotation = (
                            phase == "Rotate"
                            and qpos[1] >= 0.25
                            and "rotate_contact" not in saved_phases
                        )
                        if save_contact or save_rotation:
                            suffix = "press_contact" if save_contact else "rotate_contact"
                            cv2.imwrite(
                                str(args.output_dir / f"scene_{suffix}.png"),
                                cv2.cvtColor(scene_image, cv2.COLOR_RGB2BGR),
                            )
                            cv2.imwrite(
                                str(args.output_dir / f"tactile_{args.mode}_{suffix}.png"),
                                cv2.cvtColor(snapshot_frame.image_rgb, cv2.COLOR_RGB2BGR),
                            )
                            saved_phases.add(suffix)
            finally:
                renderer.close()
                pose_plan.close()
                sensor.close()
                backend.close()
    finally:
        writer.release()
        visualizer.close()
    if not output_video.is_file() or output_video.stat().st_size == 0:
        raise RuntimeError(f"MP4 recording failed: {output_video}")
    return {
        "video": str(output_video),
        "metrics": str(metrics_path),
        "frames": args.frames,
        "calibration": visualizer.renderer.calibration_label,
        "marker_overlay": visualizer.renderer.marker_label,
    }


def run_interactive(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the letter press HydroShear example requires CUDA")
    import mujoco
    import mujoco.viewer

    visualizer = create_visualizer(
        "hydroshear",
        mode=args.mode,
        profile=args.profile,
        calibration_dir=args.calibration_dir,
        display_gain=args.display_gain,
    )
    state = {"paused": False, "restart": False, "single_step": False}

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (ord("r"), ord("R")):
            state["restart"] = True
        elif keycode in (ord("n"), ord("N"), ord(".")):
            state["single_step"] = True
        elif keycode in (ord("1"), ord("2"), ord("3")):
            visualizer.set_mode(("rgb", "displacement", "depth")[keycode - ord("1")])

    with _scene_files() as xml_path:
        backend, sensor, pose_plan, initial_qpos, initial_qvel = _build_simulation(
            xml_path, args.device
        )
        model = backend.model
        # The render twin preserves full scene materials; every displayed state
        # is copied from the UniSim backend and this model is never stepped.
        render_model = mujoco.MjModel.from_xml_path(str(xml_path))
        render_data = mujoco.MjData(render_model)
        viewer: mujoco.viewer.Handle | None = None
        try:
            _reset(backend, sensor, initial_qpos, initial_qvel)
            viewer = mujoco.viewer.launch_passive(
                render_model,
                render_data,
                key_callback=key_callback,
                show_left_ui=False,
                show_right_ui=False,
            )
            assert viewer is not None
            with viewer.lock():
                camera_id = render_model.camera("overview").id
                camera_position = render_model.cam_pos[camera_id]
                camera_rotation = render_model.cam_mat0[camera_id].reshape(3, 3)
                camera_offset = camera_rotation[:, 2]
                default_distance = 1.75 * float(render_model.stat.extent)
                camera_distance = default_distance * np.tan(
                    np.deg2rad(render_model.cam_fovy[camera_id] / 2)
                ) / np.tan(np.deg2rad(render_model.vis.global_.fovy / 2))
                camera_distance = max(float(camera_distance), 1e-3)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = camera_position - camera_offset * camera_distance
                viewer.cam.distance = camera_distance
                viewer.cam.azimuth = float(
                    np.degrees(np.arctan2(camera_offset[1], camera_offset[0])) % 360.0
                )
                viewer.cam.elevation = float(
                    -np.degrees(np.arctan2(camera_offset[2], np.linalg.norm(camera_offset[:2])))
                )
            simulation_steps = 0
            frame_index = 0
            last_sensor_cycle = 0
            frame_time = 1.0 / args.fps
            while viewer.is_running() and (args.frames is None or frame_index < args.frames):
                now = time.perf_counter()
                if not state["paused"] or state["single_step"]:
                    sensor_cycle = frame_index // _CYCLE_FRAMES
                    if sensor_cycle > last_sensor_cycle:
                        sensor.reset()
                        last_sensor_cycle = sensor_cycle
                    phase, target = _phase_target(frame_index)
                    substeps = 1 if state["single_step"] else SUBSTEPS_PER_FRAME
                    for _ in range(substeps):
                        _one_physics_step(backend, sensor, pose_plan, target)
                        simulation_steps += 1
                    if not state["single_step"]:
                        frame_index += 1
                    state["single_step"] = False
                else:
                    phase, _ = _phase_target(max(frame_index - 1, 0))
                if state["restart"]:
                    _reset(backend, sensor, initial_qpos, initial_qvel)
                    simulation_steps = 0
                    frame_index = 0
                    last_sensor_cycle = 0
                    state["restart"] = False
                    phase = "Approach"
                tactile_frame = visualizer.update(sensor, env_id=0)
                contact_pixels = int(np.count_nonzero(tactile_frame.depth_mm < -0.05))
                # MuJoCo's passive viewer renders this model and data on its
                # own thread, so state copies and mj_forward must share its
                # lock with the render thread.
                with viewer.lock():
                    _sync_render_data(render_model, render_data, backend, simulation_steps)
                viewport = viewer.viewport
                if viewport is not None:
                    width = min(320, max(1, viewport.width - 10))
                    height = int(
                        round(
                            width
                            * tactile_frame.image_rgb.shape[0]
                            / tactile_frame.image_rgb.shape[1]
                        )
                    )
                    import cv2

                    panel = cv2.resize(
                        tactile_frame.image_rgb, (width, height), interpolation=cv2.INTER_AREA
                    )
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
                            f"Stage: {phase}",
                            f"Time: {simulation_steps * float(model.opt.timestep):.2f}s",
                        ),
                        (
                            font,
                            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                            f"Mode: {visualizer.mode}",
                            f"Contact pixels: {contact_pixels}",
                        ),
                        (
                            font,
                            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                            tactile_frame.calibration_label,
                            tactile_frame.marker_label,
                        ),
                        (
                            font,
                            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                            f"Peak depth: {tactile_frame.peak_depth_mm:.3f} mm",
                            f"Peak marker: {tactile_frame.peak_displacement_mm:.3f} mm",
                        ),
                    ]
                )
                viewer.sync()
                elapsed = time.perf_counter() - now
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
        finally:
            if viewer is not None:
                if viewer.is_running():
                    viewer.close()
                # launch_passive owns a daemon render thread. Wait for its
                # model to be released before GLFW's process-exit cleanup.
                while viewer.m is not None:
                    time.sleep(0.01)
            pose_plan.close()
            sensor.close()
            backend.close()
            visualizer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="record an MP4 instead of opening the viewer"
    )
    parser.add_argument(
        "--frames", type=int, help="maximum displayed/recorded frames; headless defaults to 300"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="CUDA device for HydroShear, e.g. cuda:0"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/letter_press"))
    parser.add_argument(
        "--profile",
        choices=("taxim", "gelsight_mini"),
        default="gelsight_mini",
        help="calibrated image profile; bundled HydroShear GelSight Mini is the default",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="optional GelSight Mini calibration override; requires --profile gelsight_mini",
    )
    parser.add_argument("--mode", choices=("rgb", "displacement", "depth"), default="rgb")
    parser.add_argument(
        "--display-gain",
        type=float,
        default=10.0,
        help="visual arrow scale; does not change raw sensor displacement",
    )
    parser.add_argument(
        "--fps", type=float, default=FRAME_RATE, help="interactive viewer update rate"
    )
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be a positive integer")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.profile == "taxim" and args.calibration_dir is not None:
        parser.error("--calibration-dir requires --profile gelsight_mini")
    if args.headless:
        args.frames = 300 if args.frames is None else args.frames
        print(json.dumps(run_headless(args), indent=2))
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
