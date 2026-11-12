"""Run the letter press HydroShear example with UniSim's Motrix backend."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from unisim.backend.base import CameraCfg, RenderClosedError
from unisim.backend.process_device import configure_backend_process_device
from unisim.factory import create_backend
from unisim.scene import SceneCfg

if __package__:
    from .display import TactileDisplay
    from .runtime import run_no_render
    from .simulation import (
        CYCLE_FRAMES,
        SIM_DT,
        LetterPressSimulation,
        scene_files,
        phase_target,
    )
else:
    from display import TactileDisplay
    from runtime import run_no_render
    from simulation import (
        CYCLE_FRAMES,
        SIM_DT,
        LetterPressSimulation,
        scene_files,
        phase_target,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-render", action="store_true", help="run physics and touch only")
    parser.add_argument("--frames", type=int, help="frames to run; viewer defaults to unlimited")
    parser.add_argument("--device", default="cuda:0", help="HydroShear CUDA device")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/letter_press_motrix"))
    parser.add_argument("--mode", choices=("rgb", "depth"), default="rgb")
    parser.add_argument("--calibration-dir", type=Path, help="GelSight Mini calibration override")
    parser.add_argument("--display-gain", type=float, default=40.0)
    parser.add_argument("--fps", type=float, default=30.0, help="viewer update rate")
    return parser


def _run_viewer(
    simulation: LetterPressSimulation,
    *,
    frames: int | None,
    fps: float,
    mode: str,
    calibration_dir: Path | None,
    display_gain: float,
) -> int:
    from motrixsim.render import Layout

    display = TactileDisplay(
        mode=mode,
        calibration_dir=calibration_dir,
        display_gain=display_gain,
    )
    backend = simulation.backend
    try:
        backend.bind_renderer(
            camera_kwargs=CameraCfg(
                cam_distance=0.11,
                cam_elevation=-25.0,
                cam_azimuth=45.0,
                cam_lookat=(0.0, 0.0, 0.008),
            )
        )
        render_app = backend._render_app
        image = None

        frame_index = 0
        last_frame = time.perf_counter()
        while frames is None or frame_index < frames:
            if frame_index and frame_index % CYCLE_FRAMES == 0:
                simulation.reset()
            phase, target = phase_target(frame_index)
            stats = simulation.advance_frame(target, capture_state=False)
            tactile = display.update(
                simulation,
                phase=phase,
                sim_time=stats["simulation_time_s"],
            )
            if image is None:
                image = render_app.create_image(tactile.panel_rgb, is_srgb=True, keep_in_cpu=True)
                render_app.widgets.create_image_widget(
                    image,
                    layout=Layout(right="0px", top="0px", width="320px", height="720px"),
                )
            else:
                image.pixels = tactile.panel_rgb
            try:
                backend.render()
            except RenderClosedError:
                break
            frame_index += 1
            elapsed = time.perf_counter() - last_frame
            delay = 1.0 / fps - elapsed
            if delay > 0:
                time.sleep(delay)
            last_frame = time.perf_counter()
        return frame_index
    finally:
        display.close()


def main() -> None:
    args = _parser().parse_args()
    if args.frames is not None and args.frames < 1:
        raise SystemExit("--frames must be a positive integer")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if not math.isfinite(args.display_gain) or args.display_gain < 0:
        raise SystemExit("--display-gain must be finite and nonnegative")
    if args.no_render and args.frames is None:
        args.frames = CYCLE_FRAMES

    with scene_files() as xml_path:
        configure_backend_process_device("motrix", args.device)
        backend = create_backend(
            "motrix",
            SceneCfg(model_file=str(xml_path)),
            num_envs=1,
            sim_dt=SIM_DT,
            base_name="indenter",
            body_state_required=True,
        )
        with LetterPressSimulation(backend, xml_path, args.device) as simulation:
            if args.no_render:
                result = run_no_render(
                    simulation,
                    frames=args.frames,
                    output_dir=args.output_dir,
                )
            else:
                print(json.dumps({"backend": "motrix", "viewer": "Motrix"}))
                frames = _run_viewer(
                    simulation,
                    frames=args.frames,
                    fps=args.fps,
                    mode=args.mode,
                    calibration_dir=args.calibration_dir,
                    display_gain=args.display_gain,
                )
                result = {"backend": "motrix", "frames": frames, "rendered": True}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
