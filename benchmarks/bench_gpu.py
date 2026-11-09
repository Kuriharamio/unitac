"""Measure HydroShear CUDA latency, throughput, transfers, and steady-state memory."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
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
from tests.helpers import (  # noqa: E402
    box_triangles,
    full_body_poses,
    hydro_config,
    make_scene,
)
from unisim.factory import create_tactile_sensor  # noqa: E402

GEL = box_triangles((0.010, 0.012625, 0.003))
GEL[:, :, 2] -= 0.003


def _poses(num_envs: int, profile: str) -> tuple[Any, Any]:
    object_positions = np.zeros((num_envs, 1, 3), dtype=np.float32)
    object_positions[..., 2] = 0.001
    positions, quaternions = full_body_poses(num_envs, object_positions)
    if profile == "torch_cuda":
        return (
            torch.as_tensor(positions, device="cuda:0"),
            torch.as_tensor(quaternions, device="cuda:0"),
        )
    if profile == "torch_cpu":
        return torch.from_numpy(positions), torch.from_numpy(quaternions)
    return positions, quaternions


def _step(sensor: Any, positions: Any, quaternions: Any, profile: str) -> None:
    if profile == "torch_cuda":
        positions[:, 0, 0].add_(1e-6)
    else:
        positions[:, 0, 0] += 1e-6
    sensor.update_pose(positions, quaternions)
    sensor.advance()


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


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _measure_case(
    num_envs: int, profile: str, warmup: int, steps: int, long_run_steps: int
) -> dict[str, Any]:
    sensor = create_tactile_sensor(
        "unitac",
        model="hydroshear",
        num_envs=num_envs,
        config=hydro_config(GEL),
    )
    sensor.materialize(make_scene())
    positions, quaternions = _poses(num_envs, profile)

    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        _step(sensor, positions, quaternions, profile)
    torch.cuda.synchronize()
    warm_memory = _memory()

    wall_start = time.perf_counter()
    for _ in range(steps):
        _step(sensor, positions, quaternions, profile)
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - wall_start
    output = sensor.read()["displacement"]
    output_pointer = output.data_ptr()
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("HydroShear output contains non-finite values")
    measured_memory = _memory()

    sensor.enable_stage_timing(True)
    latency_steps = min(steps, 100)
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    profile_start = time.perf_counter()
    for _ in range(latency_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _step(sensor, positions, quaternions, profile)
        end.record()
        event_pairs.append((start, end))
        if sensor.read()["displacement"].data_ptr() != output_pointer:
            raise RuntimeError("HydroShear changed the output buffer address during advance")
    torch.cuda.synchronize()
    profile_seconds = time.perf_counter() - profile_start
    step_ms = [start.elapsed_time(end) for start, end in event_pairs]
    output = sensor.read()["displacement"]
    if output.data_ptr() != output_pointer:
        raise RuntimeError("HydroShear changed the output buffer address during advance")
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("HydroShear output contains non-finite values")
    result: dict[str, Any] = {
        "num_envs": num_envs,
        "pads_per_env": 2,
        "markers_per_pad": [9, 7],
        "surface_points_per_object": 2048,
        "pose_profile": profile,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "wall_seconds": elapsed_seconds,
        "environment_steps_per_second": num_envs * steps / elapsed_seconds,
        "wall_ms_per_step": elapsed_seconds * 1000.0 / steps,
        "cuda_event_ms_per_step": {
            "mean": statistics.fmean(step_ms),
            "p50": float(np.percentile(step_ms, 50)),
            "p95": float(np.percentile(step_ms, 95)),
            "max": max(step_ms),
        },
        "latency_profile_steps": latency_steps,
        "latency_profile_wall_seconds": profile_seconds,
        "stage_ms_last_step": dict(sensor.stage_timings_ms),
        "transfer_diagnostics": dict(sensor.transfer_diagnostics),
        "memory_after_warmup": warm_memory,
        "memory_after_measurement": measured_memory,
        "memory_delta_during_measurement": {
            name: measured_memory[name] - warm_memory[name] for name in warm_memory
        },
        "output_finite": True,
        "output_buffer_address_stable": True,
    }

    if long_run_steps:
        sensor.enable_stage_timing(False)
        before = _memory()
        long_start = time.perf_counter()
        for _ in range(long_run_steps):
            _step(sensor, positions, quaternions, profile)
        torch.cuda.synchronize()
        long_seconds = time.perf_counter() - long_start
        after = _memory()
        output = sensor.read()["displacement"]
        finite = bool(torch.isfinite(output).all().item())
        pointer_stable = output.data_ptr() == output_pointer
        delta = {name: after[name] - before[name] for name in before}
        stable_memory = all(
            delta[name] == 0 for name in ("torch_allocated_bytes", "cupy_pool_used_bytes")
        )
        result["long_run"] = {
            "steps": long_run_steps,
            "wall_seconds": long_seconds,
            "environment_steps_per_second": num_envs * long_run_steps / long_seconds,
            "memory_before": before,
            "memory_after": after,
            "memory_delta": delta,
            "output_finite": finite,
            "output_buffer_address_stable": pointer_stable,
            "live_device_memory_stable_after_warmup": stable_memory,
        }
        if not finite or not pointer_stable or not stable_memory:
            raise RuntimeError(f"long-run stability check failed: {result['long_run']}")

    sensor.close()
    del sensor, positions, quaternions
    gc.collect()
    torch.cuda.synchronize()
    result["memory_after_close"] = _memory()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", type=int, default=[1, 64, 1024])
    parser.add_argument(
        "--pose-profiles",
        nargs="+",
        choices=("numpy", "torch_cpu", "torch_cuda"),
        default=["numpy", "torch_cuda"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--long-run-steps", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/hydroshear_cuda.json"),
    )
    args = parser.parse_args()
    if any(count <= 0 for count in args.envs):
        parser.error("environment counts must be positive")
    if args.warmup < 0 or args.steps <= 0 or args.long_run_steps < 0:
        parser.error("warmup and long-run steps must be non-negative; steps must be positive")
    if not torch.cuda.is_available():
        parser.error("HydroShear benchmark requires a working CUDA device")

    cases = []
    long_run_assigned = False
    for num_envs in args.envs:
        for profile in args.pose_profiles:
            long_run_steps = args.long_run_steps if num_envs == 64 and not long_run_assigned else 0
            case = _measure_case(
                num_envs,
                profile,
                args.warmup,
                args.steps,
                long_run_steps,
            )
            if long_run_steps:
                long_run_assigned = True
            cases.append(case)
            print(
                f"profile={profile} envs={num_envs} "
                f"env/s={case['environment_steps_per_second']:.1f} "
                f"step_ms={case['wall_ms_per_step']:.3f}"
            )
    if args.long_run_steps and 64 not in args.envs:
        raise ValueError("include --envs 64 to run the requested long-run stability check")

    result = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cupy": cp.__version__,
        "solver": {"use_fft_dilation": False, "use_matmul_shear": True},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
