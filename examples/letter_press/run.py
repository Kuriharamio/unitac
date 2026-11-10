"""Run the letter press HydroShear example on a selected UniSim backend."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from unisim import ADAPTER_SPECS

if __package__:
    from .simulation import (
        CYCLE_FRAMES,
        LetterPressSimulation,
        observation_metrics,
        phase_target,
        scene_files,
    )
else:
    from simulation import (
        CYCLE_FRAMES,
        LetterPressSimulation,
        observation_metrics,
        phase_target,
        scene_files,
    )


def _write_reset_record(simulation: LetterPressSimulation, metrics_file: Any) -> None:
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
        "target": [0.0, 0.0],
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


def run_no_render(
    simulation: LetterPressSimulation, *, frames: int, output_dir: Path
) -> dict[str, Any]:
    """Advance physics and touch without importing or creating renderers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        _write_reset_record(simulation, metrics_file)
        for frame_index in range(frames):
            if frame_index and frame_index % CYCLE_FRAMES == 0:
                _write_reset_record(simulation, metrics_file)
            phase, target = phase_target(frame_index)
            total_started = time.perf_counter()
            stats = simulation.advance_frame(target)
            diagnostics_started = time.perf_counter()
            assert simulation.sensor is not None
            snapshot = simulation.sensor.read_visualization_snapshot(env_id=0)
            observed = observation_metrics(
                snapshot.displacement[0], snapshot.depth_mm[0], snapshot.generation
            )
            diagnostics_ms = (time.perf_counter() - diagnostics_started) * 1000.0
            record = {
                "backend": simulation.backend_name,
                "phase": phase,
                "simulation_time_s": stats["simulation_time_s"],
                "target": target.tolist(),
                "qpos": stats["qpos"].tolist(),
                "qvel": stats["qvel"].tolist(),
                **observed,
                "physics_ms": stats["physics_ms"],
                "pose_read_transfer_ms": stats["pose_read_transfer_ms"],
                "tactile_ms": stats["tactile_ms"],
                "diagnostic_transfer_ms": diagnostics_ms,
                "visualization_ms": 0.0,
                "total_step_ms": (time.perf_counter() - total_started) * 1000.0,
                "backend_step_timing_ms": stats["backend_timing"],
                "transfers": simulation.transfer_diagnostics(),
            }
            metrics_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        _write_reset_record(simulation, metrics_file)
    return {
        "backend": simulation.backend_name,
        "frames": frames,
        "rendered": False,
        "metrics": str(metrics_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=tuple(spec.name for spec in ADAPTER_SPECS),
        default="mujoco",
        help="UniSim physics backend",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--headless", action="store_true", help="record a rendered MP4")
    output.add_argument("--no-render", action="store_true", help="run physics and touch only")
    parser.add_argument("--frames", type=int, help="frames to run; GUI defaults to unlimited")
    parser.add_argument("--device", default="cuda:0", help="HydroShear CUDA device")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/letter_press"))
    parser.add_argument("--mode", choices=("rgb", "depth"), default="rgb")
    parser.add_argument("--calibration-dir", type=Path, help="GelSight Mini calibration override")
    parser.add_argument("--display-gain", type=float, default=40.0)
    parser.add_argument("--fps", type=float, default=30.0, help="GUI and recording frame rate")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.frames is not None and args.frames < 1:
        raise SystemExit("--frames must be a positive integer")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.display_gain < 0 or not np.isfinite(args.display_gain):
        raise SystemExit("--display-gain must be finite and nonnegative")
    if args.headless or args.no_render:
        args.frames = CYCLE_FRAMES if args.frames is None else args.frames

    with scene_files() as xml_path:
        with LetterPressSimulation(args.backend, xml_path, args.device) as simulation:
            if args.no_render:
                result = run_no_render(
                    simulation,
                    frames=args.frames,
                    output_dir=args.output_dir,
                )
            elif args.headless:
                if __package__:
                    from .viewer import run_headless
                else:
                    from viewer import run_headless
                result = run_headless(
                    simulation,
                    frames=args.frames,
                    output_dir=args.output_dir,
                    mode=args.mode,
                    fps=args.fps,
                    calibration_dir=args.calibration_dir,
                    display_gain=args.display_gain,
                )
            else:
                if __package__:
                    from .viewer import run_interactive
                else:
                    from viewer import run_interactive
                print(json.dumps({"backend": simulation.backend_name, "viewer": "MuJoCo playback"}))
                run_interactive(
                    simulation,
                    frames=args.frames,
                    fps=args.fps,
                    mode=args.mode,
                    calibration_dir=args.calibration_dir,
                    display_gain=args.display_gain,
                )
                return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
