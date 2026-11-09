"""Compare the pinned HydroShear batch API with UniTac on identical CUDA cases."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tests.helpers import box_triangles, full_body_poses, hydro_config, make_scene  # noqa: E402
from unisim.factory import create_tactile_sensor  # noqa: E402

REFERENCE_COMMIT = "26403177779c4f74689761c041fa9580b075dc08"
GEL = box_triangles((0.010, 0.012625, 0.003))
GEL[:, :, 2] -= 0.003
OBJECT = box_triangles((0.005, 0.005, 0.004))


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _load_reference(reference_repo: Path) -> Any:
    revision = _git_revision(reference_repo)
    if revision != REFERENCE_COMMIT:
        raise ValueError(
            f"reference repository must be {REFERENCE_COMMIT}, got {revision}"
        )
    sys.path.insert(0, str(reference_repo / "src"))
    import hydroshear.batch as legacy_batch

    # Both implementations receive the same synthetic, closed gel mesh; no
    # calibration assets from the reference repository enter this comparison.
    legacy_batch.gelsight_mini_gel_triangles = lambda: GEL
    return legacy_batch.HydroShearBatch


def _poses(num_envs: int, profile: str) -> tuple[Any, Any]:
    objects = np.zeros((num_envs, 1, 3), dtype=np.float32)
    objects[..., 2] = 0.001
    positions, quaternions = full_body_poses(num_envs, objects)
    if profile == "torch_cuda":
        return (
            torch.as_tensor(positions, device="cuda:0"),
            torch.as_tensor(quaternions, device="cuda:0"),
        )
    return positions, quaternions


def _move(positions: Any, profile: str) -> None:
    if profile == "torch_cuda":
        positions[:, 0, 0].add_(1e-6)
    else:
        positions[:, 0, 0] += 1e-6


def _memory() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    pool = cp.get_default_memory_pool()
    return {
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cupy_pool_used_bytes": int(pool.used_bytes()),
        "cupy_pool_total_bytes": int(pool.total_bytes()),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
    }


def _result(
    name: str,
    num_envs: int,
    profile: str,
    warmup: int,
    steps: int,
    elapsed: float,
    before: dict[str, int],
    after: dict[str, int],
    output: torch.Tensor,
    pointer: int,
    transfers: dict[str, int],
    stage_times: dict[str, float] | None,
) -> dict[str, Any]:
    return {
        "implementation": name,
        "num_envs": num_envs,
        "pads_per_env": 2,
        "markers_per_pad": [9, 7],
        "surface_points_per_object": 2048,
        "pose_profile": profile,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "environment_steps_per_second": num_envs * steps / elapsed,
        "wall_ms_per_step": elapsed * 1000.0 / steps,
        "stage_ms_last_step": stage_times,
        "transfer_diagnostics": transfers,
        "memory_before_measurement": before,
        "memory_after_measurement": after,
        "memory_delta_during_measurement": {
            key: after[key] - before[key] for key in before
        },
        "output_finite": bool(torch.isfinite(output).all().item()),
        "output_buffer_address_stable": output.data_ptr() == pointer,
    }


def _run_new(
    num_envs: int, profile: str, warmup: int, steps: int
) -> tuple[dict[str, Any], torch.Tensor]:
    sensor = create_tactile_sensor(
        "unitac",
        model="hydroshear",
        num_envs=num_envs,
        config=hydro_config(GEL),
    )
    sensor.materialize(make_scene())
    positions, quaternions = _poses(num_envs, profile)

    def step(
        current_sensor: Any, current_positions: Any, current_quaternions: Any
    ) -> None:
        _move(current_positions, profile)
        current_sensor.update_pose(current_positions, current_quaternions)
        current_sensor.advance()

    for _ in range(warmup):
        step(sensor, positions, quaternions)
    torch.cuda.synchronize()
    output = sensor.read()["displacement"]
    pointer = output.data_ptr()
    torch.cuda.reset_peak_memory_stats()
    sensor.enable_stage_timing(False)
    before = _memory()
    start = time.perf_counter()
    for _ in range(steps):
        step(sensor, positions, quaternions)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    output = sensor.read()["displacement"]
    after = _memory()
    output_snapshot = output.detach().clone()
    result = _result(
        "unitac",
        num_envs,
        profile,
        warmup,
        steps,
        elapsed,
        before,
        after,
        output,
        pointer,
        dict(sensor.transfer_diagnostics),
        None,
    )
    sensor.enable_stage_timing(True)
    step(sensor, positions, quaternions)
    torch.cuda.synchronize()
    result["stage_ms_last_step"] = dict(sensor.stage_timings_ms)
    sensor.close()
    del sensor, positions, quaternions
    gc.collect()
    torch.cuda.synchronize()
    return result, output_snapshot


def _run_reference(
    sensor_type: Any, num_envs: int, profile: str, warmup: int, steps: int
) -> tuple[dict[str, Any], torch.Tensor]:
    config = {
        "pad_count": 2,
        "channels": ["dx", "dy", "dz"],
        "surface_points": 2048,
        "output_grid": {
            "rows": 9,
            "cols": 7,
            "spacing_mm": [2.145, 2.153125],
            "center_offset_mm": [-0.3575, -0.325],
            "pixel_scale_mm": 0.065,
        },
        "solver": {"use_fft_dilation": False, "use_matmul_shear": True},
        "device": "cuda:0",
        "download_state": False,
    }
    sensor = sensor_type([OBJECT], num_envs, config)
    output_binding = sensor.bind_output()
    positions, quaternions = _poses(num_envs, profile)
    object_pos, pad_pos = positions[:, :1], positions[:, 1:]
    object_quat, pad_quat = quaternions[:, :1], quaternions[:, 1:]
    device = "cuda:0" if profile == "torch_cuda" else None

    def step(current_positions: Any, current_sensor: Any, binding: Any) -> None:
        _move(current_positions, profile)
        current_sensor.step(
            object_pos,
            object_quat,
            pad_pos,
            pad_quat,
            device=device,
            out=binding,
        )

    for _ in range(warmup):
        step(positions, sensor, output_binding)
    torch.cuda.synchronize()
    output = output_binding.read()["displacement"]
    pointer = output.data_ptr()
    torch.cuda.reset_peak_memory_stats()
    before = _memory()
    start = time.perf_counter()
    for _ in range(steps):
        step(positions, sensor, output_binding)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    output = output_binding.read()["displacement"]
    after = _memory()
    result = _result(
        "hydroshear_reference",
        num_envs,
        profile,
        warmup,
        steps,
        elapsed,
        before,
        after,
        output,
        pointer,
        dict(sensor.transfer_diagnostics()),
        None,
    )
    del sensor, output_binding, positions, quaternions
    gc.collect()
    torch.cuda.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    return result, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--envs", nargs="+", type=int, default=[1, 64, 1024])
    parser.add_argument(
        "--pose-profiles",
        nargs="+",
        choices=("numpy", "torch_cuda"),
        default=["numpy", "torch_cuda"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/hydroshear_reference_comparison.json"),
    )
    args = parser.parse_args()
    if any(count <= 0 for count in args.envs) or args.warmup < 0 or args.steps <= 0:
        parser.error("environment counts and steps must be positive; warmup non-negative")
    if not torch.cuda.is_available():
        parser.error("HydroShear comparison requires a working CUDA device")

    sensor_type = _load_reference(args.reference_repo.expanduser().resolve())
    cases = []
    for num_envs in args.envs:
        for profile in args.pose_profiles:
            torch.cuda.empty_cache()
            reference, reference_output = _run_reference(
                sensor_type, num_envs, profile, args.warmup, args.steps
            )
            current, current_output = _run_new(num_envs, profile, args.warmup, args.steps)
            difference = current_output - reference_output
            max_abs_diff = float(difference.abs().max().item())
            relative_l2 = float(
                (torch.linalg.vector_norm(difference)
                 / torch.linalg.vector_norm(reference_output).clamp_min(1e-12)).item()
            )
            numerically_close = bool(
                torch.allclose(current_output, reference_output, rtol=2e-6, atol=2e-5)
            )
            if not numerically_close:
                raise RuntimeError(
                    f"reference output differs for envs={num_envs}, profile={profile}: "
                    f"max_abs={max_abs_diff} mm"
                )
            cases.append(
                {
                    "num_envs": num_envs,
                    "pose_profile": profile,
                    "reference": reference,
                    "unitac": current,
                    "max_abs_output_difference_mm": max_abs_diff,
                    "relative_l2_output_difference": relative_l2,
                    "outputs_match_reference_tolerance": numerically_close,
                }
            )
            print(
                f"envs={num_envs} profile={profile} "
                f"reference_env/s={reference['environment_steps_per_second']:.1f} "
                f"unitac_env/s={current['environment_steps_per_second']:.1f} "
                f"max_abs_diff_mm={max_abs_diff:.3g}"
            )
            del reference_output, current_output, difference
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()

    result = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "reference_repository_commit": REFERENCE_COMMIT,
        "unitac_repository_commit": _git_revision(REPO),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cupy": cp.__version__,
        "solver": {"use_fft_dilation": False, "use_matmul_shear": True},
        "comparison_scope": (
            "Steady-state end-to-end public API. HydroShearBatch.step uses an explicit "
            "bound output copy; UniTac uses update_pose/advance/read with its owned "
            "buffer. Timings include each API path and do not isolate solver-only speed."
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
