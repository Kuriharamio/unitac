"""Benchmark HydroShear at representative mesh, contact, and batch sizes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import resources
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
    gelsight_collision_mesh,
    hydro_config,
    make_scene,
)
from unisim.factory import create_tactile_sensor  # noqa: E402

from unitac.hydroshear.fft_shear import shear_matmul, shear_matmul_active  # noqa: E402

SYNTHETIC_GEL = box_triangles((0.010, 0.012625, 0.003))
SYNTHETIC_GEL[:, :, 2] -= 0.003
GELSIGHT_MESH_RESOURCE = resources.files("unitac.hydroshear").joinpath(
    "assets", "letter_press", "gsmini_elastomer_collision_512.obj"
)
GELSIGHT_OFFSET = np.asarray([0.0005, 0.05975, 0.00725], dtype=np.float32)
DEVICE = torch.device("cuda:0")


def _gel(name: str) -> tuple[np.ndarray, dict[str, Any]]:
    if name == "synthetic12":
        return SYNTHETIC_GEL, {"triangles": 12, "source": "closed synthetic box"}
    mesh = gelsight_collision_mesh().copy() + GELSIGHT_OFFSET
    digest = hashlib.sha256(GELSIGHT_MESH_RESOURCE.read_bytes()).hexdigest()
    return mesh, {
        "triangles": int(mesh.shape[0]),
        "source": "HydroShear MIT asset gsmini_elastomer_collision_512.obj",
        "source_path": "unitac.hydroshear/assets/letter_press/gsmini_elastomer_collision_512.obj",
        "sha256": digest,
        "sensor_frame_offset_m": GELSIGHT_OFFSET.tolist(),
    }


def _poses(num_envs: int, profile: str, scenario: str) -> tuple[Any, Any]:
    object_positions = np.zeros((num_envs, 1, 3), dtype=np.float32)
    object_positions[..., 2] = 0.020 if scenario == "no_contact" else 0.001
    positions, quaternions = full_body_poses(num_envs, object_positions)
    if profile == "torch_cuda":
        return (
            torch.as_tensor(positions, device=DEVICE),
            torch.as_tensor(quaternions, device=DEVICE),
        )
    if profile == "torch_cpu":
        return torch.from_numpy(positions), torch.from_numpy(quaternions)
    return positions, quaternions


def _step(sensor: Any, positions: Any, quaternions: Any, profile: str, scenario: str) -> None:
    if scenario == "press":
        if profile == "torch_cuda":
            positions[:, 0, 2].sub_(1e-5)
        else:
            positions[:, 0, 2] -= 1e-5
    elif scenario == "slide":
        if profile == "torch_cuda":
            positions[:, 0, 0].add_(1e-5)
            positions[:, 0, 2].sub_(1e-5)
        else:
            positions[:, 0, 0] += 1e-5
            positions[:, 0, 2] -= 1e-5
    sensor.update_pose(positions, quaternions)
    sensor.advance()


def _memory() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(DEVICE)
    pool = cp.get_default_memory_pool()
    return {
        "torch_allocated_bytes": int(torch.cuda.memory_allocated(DEVICE)),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved(DEVICE)),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
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


def _profiler_report(profiler: Any) -> dict[str, Any]:
    rows = []
    for event in profiler.key_averages():
        name = event.key
        key = name.lower()
        if not any(
            token in key
            for token in (
                "memcpy",
                "copy_",
                "cuda memcpy",
                "synchron",
                "aten::item",
                "local_scalar",
                "nonzero",
                "aten::index",
            )
        ):
            continue
        cuda_us = getattr(event, "device_time_total", None)
        if cuda_us is None:
            cuda_us = getattr(event, "cuda_time_total", 0.0)
        rows.append(
            {
                "name": name,
                "count": int(event.count),
                "cpu_time_us": float(event.cpu_time_total),
                "cuda_time_us": float(cuda_us or 0.0),
                "device_memory_bytes": int(getattr(event, "device_memory_usage", 0)),
            }
        )
    h2d = sum(
        row["count"]
        for row in rows
        if any(token in row["name"].lower() for token in ("htod", "host to device"))
    )
    d2h = sum(
        row["count"]
        for row in rows
        if any(token in row["name"].lower() for token in ("dtoh", "device to host"))
    )
    sync = sum(
        row["count"]
        for row in rows
        if any(
            token in row["name"].lower()
            for token in ("synchron", "aten::item", "local_scalar", "nonzero")
        )
    )
    sync_apis = [
        {"name": row["name"], "count": row["count"]}
        for row in rows
        if any(token in row["name"].lower() for token in ("synchronize", "streamwait"))
    ]
    dynamic_shape_events = sum(
        row["count"]
        for row in rows
        if any(token in row["name"].lower() for token in ("aten::item", "local_scalar", "nonzero"))
    )
    return {
        "host_to_device_copy_events": h2d,
        "device_to_host_copy_events": d2h,
        "synchronization_api_events": sync_apis,
        "scalar_or_dynamic_shape_sync_events": dynamic_shape_events,
        "all_sync_or_dynamic_index_event_count": sync,
        "relevant_events": rows,
        "profiler_capture_boundary_sync_included": True,
    }


def _capture_profile(
    sensor: Any, positions: Any, quaternions: Any, profile: str, scenario: str
) -> dict[str, Any]:
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
    )
    with profiler:
        _step(sensor, positions, quaternions, profile, scenario)
        output = sensor.read()["displacement"]
        if output.device != DEVICE:
            raise RuntimeError(f"normal read returned {output.device}, expected {DEVICE}")
    torch.cuda.synchronize(DEVICE)
    result = _profiler_report(profiler)
    result["read_device"] = str(output.device)
    return result


def _active_point_stats(sensor: Any, num_envs: int) -> dict[str, float | int]:
    forces = sensor._solver.hydrosoft_forces.reshape(num_envs, 2, -1, 3)
    counts = forces.abs().amax(dim=-1).gt(0).sum(dim=-1).reshape(-1)
    values = counts.detach().cpu().numpy()
    return {
        "per_pad_env_min": int(values.min()),
        "per_pad_env_mean": float(values.mean()),
        "per_pad_env_max": int(values.max()),
        "active_fraction_of_surface_samples": float(values.mean() / forces.shape[2]),
    }


def _measure_case(
    gel_name: str,
    gel_mesh: np.ndarray,
    gel_metadata: dict[str, Any],
    scenario: str,
    num_envs: int,
    profile: str,
    warmup: int,
    steps: int,
    long_run_steps: int,
) -> dict[str, Any]:
    sensor = create_tactile_sensor(
        tactile_type="hydroshear",
        num_envs=num_envs,
        config=hydro_config(gel_mesh),
    )
    sensor.materialize(make_scene())
    positions, quaternions = _poses(num_envs, profile, scenario)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    for _ in range(warmup):
        _step(sensor, positions, quaternions, profile, scenario)
    torch.cuda.synchronize(DEVICE)
    warm_memory = _memory()

    wall_start = time.perf_counter()
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(DEVICE))
        _step(sensor, positions, quaternions, profile, scenario)
        end.record(torch.cuda.current_stream(DEVICE))
        event_pairs.append((start, end))
    torch.cuda.synchronize(DEVICE)
    wall_seconds = time.perf_counter() - wall_start
    elapsed_ms = [start.elapsed_time(end) for start, end in event_pairs]
    output = sensor.read()["displacement"]
    output_pointer = output.data_ptr()
    output_finite = bool(torch.isfinite(output).all().item())
    output_norm = float(torch.linalg.vector_norm(output).item())
    max_environment_row_difference = float(
        torch.max(torch.abs(output - output[0:1])).item()
    )
    if not output_finite:
        raise RuntimeError(f"{gel_name}/{scenario}/{profile}/{num_envs}: non-finite output")
    if not torch.allclose(output, output[0:1].expand_as(output), rtol=2e-6, atol=2e-5):
        raise RuntimeError(
            f"{gel_name}/{scenario}/{profile}/{num_envs}: environment rows differ"
        )
    if scenario == "no_contact" and output_norm > 1e-5:
        raise RuntimeError(f"no-contact case produced nonzero output norm {output_norm}")

    sensor.enable_stage_timing(True)
    stage_start = torch.cuda.Event(enable_timing=True)
    stage_end = torch.cuda.Event(enable_timing=True)
    stage_start.record(torch.cuda.current_stream(DEVICE))
    _step(sensor, positions, quaternions, profile, scenario)
    stage_end.record(torch.cuda.current_stream(DEVICE))
    stage_end.synchronize()
    stage_step_ms = float(stage_start.elapsed_time(stage_end))
    stages_ms = dict(sensor.stage_timings_ms)
    transfer_profile = _capture_profile(sensor, positions, quaternions, profile, scenario)
    active_points = _active_point_stats(sensor, num_envs)
    measured_memory = _memory()
    if scenario != "no_contact" and active_points["per_pad_env_max"] == 0:
        raise RuntimeError(f"{gel_name}/{scenario} case did not create active contact points")

    result: dict[str, Any] = {
        "gel": gel_name,
        "gel_metadata": gel_metadata,
        "scenario": scenario,
        "num_envs": num_envs,
        "pads_per_env": 2,
        "markers_per_pad": [9, 7],
        "surface_points_per_object": 2048,
        "pose_profile": profile,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "wall_seconds": wall_seconds,
        "environment_steps_per_second": num_envs * steps / wall_seconds,
        "total_wall_ms_per_step": wall_seconds * 1000.0 / steps,
        "cuda_event_ms_per_step": {
            "mean": statistics.fmean(elapsed_ms),
            "p50": float(np.percentile(elapsed_ms, 50)),
            "p95": float(np.percentile(elapsed_ms, 95)),
            "max": max(elapsed_ms),
        },
        "stage_timing_step_ms": stage_step_ms,
        "stage_ms_last_step": stages_ms,
        "active_contact_points": active_points,
        "output_norm_model_units": output_norm,
        "max_environment_row_difference_model_units": max_environment_row_difference,
        "transfer_diagnostics": dict(sensor.transfer_diagnostics),
        "profiled_transfer_and_sync_events": transfer_profile,
        "memory_after_warmup": warm_memory,
        "memory_after_measurement": measured_memory,
        "peak_torch_allocated_bytes": measured_memory["torch_peak_allocated_bytes"],
        "peak_tracked_torch_plus_cupy_bytes": (
            measured_memory["torch_peak_allocated_bytes"] + measured_memory["cupy_pool_total_bytes"]
        ),
        "output_finite": output_finite,
        "output_buffer_address_stable": output.data_ptr() == output_pointer,
    }

    if long_run_steps:
        sensor.enable_stage_timing(False)
        before = _memory()
        long_start = time.perf_counter()
        for _ in range(long_run_steps):
            _step(sensor, positions, quaternions, profile, scenario)
        torch.cuda.synchronize(DEVICE)
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
    torch.cuda.synchronize(DEVICE)
    result["memory_after_close"] = _memory()
    return result


def _measure_shear_active(
    active_counts: list[int], warmup: int, steps: int, num_envs: int = 64
) -> dict[str, Any]:
    generator = torch.Generator(device=DEVICE).manual_seed(5090)
    tactile = torch.rand((num_envs, 126, 3), generator=generator, device=DEVICE)
    indenter = torch.rand((num_envs, 2048, 3), generator=generator, device=DEVICE)
    force = torch.rand((num_envs, 2048, 3), generator=generator, device=DEVICE)
    result = []
    for active_count in active_counts:
        masked_force = force.clone()
        if active_count < masked_force.shape[1]:
            masked_force[:, active_count:] = 0
        args = (tactile, indenter, masked_force, 10_800.0, 1000 / 0.065)
        reference = shear_matmul(*args)
        candidate = shear_matmul_active(*args, max_points=256)
        torch.cuda.synchronize(DEVICE)
        parity = bool(torch.allclose(reference, candidate, rtol=1e-5, atol=1e-6))
        if not parity:
            raise RuntimeError(
                f"shear_matmul_active changed output at {active_count} active points"
            )

        timings: dict[str, list[float]] = {"full_gemm": [], "active_wrapper": []}
        for name, function in (
            ("full_gemm", lambda: shear_matmul(*args)),
            ("active_wrapper", lambda: shear_matmul_active(*args, max_points=256)),
        ):
            for _ in range(warmup):
                function()
            event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            for _ in range(steps):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream(DEVICE))
                function()
                end.record(torch.cuda.current_stream(DEVICE))
                event_pairs.append((start, end))
            torch.cuda.synchronize(DEVICE)
            timings[name] = [float(start.elapsed_time(end)) for start, end in event_pairs]
        result.append(
            {
                "environments": num_envs,
                "pads_per_environment": 2,
                "tactile_points_per_environment": 126,
                "surface_points": 2048,
                "active_points_per_environment": active_count,
                "active_fraction": active_count / 2048,
                "dense_fallback_expected_on_cuda": True,
                "exact_output_parity": parity,
                "full_gemm_cuda_event_ms": {
                    "median": float(np.median(timings["full_gemm"])),
                    "p95": float(np.percentile(timings["full_gemm"], 95)),
                },
                "active_wrapper_cuda_event_ms": {
                    "median": float(np.median(timings["active_wrapper"])),
                    "p95": float(np.percentile(timings["active_wrapper"], 95)),
                },
            }
        )
    return {
        "reason_for_cuda_fallback": (
            "CUDA active-count branching would synchronize the device; full GEMM preserves "
            "exact unsampled contact values."
        ),
        "max_points": 256,
        "measurements": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", type=int, default=[1, 64, 1024])
    parser.add_argument(
        "--pose-profiles",
        nargs="+",
        choices=("numpy", "torch_cuda"),
        default=["numpy", "torch_cuda"],
    )
    parser.add_argument(
        "--gels",
        nargs="+",
        choices=("synthetic12", "gelsight512"),
        default=["synthetic12", "gelsight512"],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("no_contact", "press", "slide"),
        default=["no_contact", "press", "slide"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--long-run-steps", type=int, default=1000)
    parser.add_argument("--shear-warmup", type=int, default=20)
    parser.add_argument("--shear-steps", type=int, default=100)
    parser.add_argument("--shear-envs", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/hydroshear_phase2_gpu.json"),
    )
    args = parser.parse_args()
    if any(count <= 0 for count in args.envs):
        parser.error("environment counts must be positive")
    if args.warmup < 0 or args.steps <= 0 or args.long_run_steps < 0:
        parser.error("warmup and long-run steps must be non-negative; steps must be positive")
    if not torch.cuda.is_available():
        parser.error("HydroShear benchmark requires a working CUDA device")
    gels = {name: _gel(name) for name in args.gels}
    cases = []
    for gel_name, (gel_mesh, metadata) in gels.items():
        for scenario in args.scenarios:
            for num_envs in args.envs:
                for profile in args.pose_profiles:
                    long_run_steps = args.long_run_steps if num_envs == 64 else 0
                    case = _measure_case(
                        gel_name,
                        gel_mesh,
                        metadata,
                        scenario,
                        num_envs,
                        profile,
                        args.warmup,
                        args.steps,
                        long_run_steps,
                    )
                    cases.append(case)
                    print(
                        f"gel={gel_name} scenario={scenario} profile={profile} "
                        f"envs={num_envs} step_ms={case['total_wall_ms_per_step']:.3f}"
                    )
    if args.long_run_steps and 64 not in args.envs:
        parser.error("include --envs 64 to run the requested long-run stability check")
    shear_analysis = _measure_shear_active(
        [0, 32, 256, 1024, 2048], args.shear_warmup, args.shear_steps, args.shear_envs
    )
    result = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "gpu": torch.cuda.get_device_name(DEVICE),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cupy": cp.__version__,
        "workload": {
            "pads_per_env": 2,
            "markers_per_pad": [9, 7],
            "surface_points_per_object": 2048,
            "selector_device": str(DEVICE),
        },
        "solver": {"use_fft_dilation": False, "use_matmul_shear": True},
        "cases": cases,
        "shear_matmul_active_analysis": shear_analysis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
