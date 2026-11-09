from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch

from unitac import create_tactile_sensor

from .helpers import box_triangles, full_body_poses, hydro_config, make_scene

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="HydroShear requires CUDA")


GEL = box_triangles((0.010, 0.012625, 0.003))
GEL[:, :, 2] -= 0.003


@pytest.mark.parametrize("num_envs", [64, 1024])
@pytest.mark.parametrize("selector_kind", ["mask", "indices", "cupy_indices"])
def test_device_selectors_update_and_reset_selected_environment_rows(
    num_envs, selector_kind, monkeypatch
):
    sensor = create_tactile_sensor(
        model="hydroshear",
        num_envs=num_envs,
        config=hydro_config(GEL, surface_points=1),
    )
    sensor.materialize(make_scene())
    selected = np.arange(0, num_envs, 4, dtype=np.int64)
    selector = _selector(selector_kind, selected, num_envs)
    object_positions = np.tile(
        np.asarray([[[0.01, 0.02, 0.03]]], dtype=np.float32), (len(selected), 1, 1)
    )
    positions_np, quaternions_np = full_body_poses(len(selected), object_positions)
    positions = torch.as_tensor(positions_np, device="cuda:0")
    quaternions = torch.as_tensor(quaternions_np, device="cuda:0")
    body_pos_before = torch.from_dlpack(sensor._body_pos).clone()
    transfer_before = dict(sensor.transfer_diagnostics)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CUDA selector path attempted a host materialization")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        sensor.update_pose(positions, quaternions, selector)
        sensor._output.fill_(1.0)
        sensor._solver._reset_mask.zero_()
        sensor._solver.hydrosoft_forces.fill_(1.0)
        sensor._solver.prev_sdf.fill_(2.0)
        sensor._solver.prev_grasped_obj_pts_elastomer.fill_(3.0)
        output_address = sensor._output.data_ptr()
        sensor.reset(selector)

    torch.cuda.synchronize("cuda:0")
    body_pos_after = torch.from_dlpack(sensor._body_pos)
    expected_pos = body_pos_before.clone()
    expected_pos[selected] = positions
    torch.testing.assert_close(body_pos_after, expected_pos)

    output = sensor.read()["displacement"]
    assert output.data_ptr() == output_address
    assert torch.count_nonzero(output[selected]) == 0
    assert torch.all(output[np.setdiff1d(np.arange(num_envs), selected)] == 1.0)
    force = sensor._solver.hydrosoft_forces.reshape(num_envs, 2, 1, 3)
    assert torch.count_nonzero(force[selected]) == 0
    assert torch.all(force[np.setdiff1d(np.arange(num_envs), selected)] == 1.0)
    torch.testing.assert_close(
        sensor._solver.prev_sdf, torch.full_like(sensor._solver.prev_sdf, 2.0)
    )
    torch.testing.assert_close(
        sensor._solver.prev_grasped_obj_pts_elastomer,
        torch.full_like(sensor._solver.prev_grasped_obj_pts_elastomer, 3.0),
    )
    reset_rows = torch.zeros(num_envs, dtype=torch.bool, device="cuda:0")
    reset_rows[selected] = True
    torch.testing.assert_close(sensor._solver._reset_mask.reshape(num_envs, 2)[:, 0], reset_rows)
    assert dict(sensor.transfer_diagnostics) == transfer_before
    sensor.close()


@pytest.mark.parametrize("selector_kind", ["mask", "indices", "cupy_indices"])
def test_cuda_selector_path_has_no_profiled_host_transfer_or_scalar_sync(selector_kind):
    torch = pytest.importorskip("torch")
    torch.cuda.init()
    sensor = create_tactile_sensor(
        model="hydroshear",
        num_envs=64,
        config=hydro_config(GEL, surface_points=1),
    )
    sensor.materialize(make_scene())
    selected = np.arange(0, 64, 2, dtype=np.int64)
    selector = _selector(selector_kind, selected, 64)
    positions_np, quaternions_np = full_body_poses(
        len(selected), np.zeros((len(selected), 1, 3), dtype=np.float32)
    )
    positions = torch.as_tensor(positions_np, device="cuda:0")
    quaternions = torch.as_tensor(quaternions_np, device="cuda:0")
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    )
    with profiler:
        sensor.update_pose(positions, quaternions, selector)
        sensor.reset(selector)
    torch.cuda.synchronize("cuda:0")
    event_names = [event.name.lower() for event in profiler.events()]
    assert not any(name in ("aten::item", "aten::_local_scalar_dense") for name in event_names)
    assert not any(
        "memcpy dtoh" in name or "memcpy htod" in name or "cuda memcpy dtoh" in name
        for name in event_names
    )
    assert "aten::nonzero" not in event_names
    sensor.close()


def test_empty_device_selectors_leave_output_and_history_unchanged():
    sensor = create_tactile_sensor(
        model="hydroshear",
        num_envs=64,
        config=hydro_config(GEL, surface_points=1),
    )
    sensor.materialize(make_scene())
    sensor._output.fill_(3.0)
    sensor._solver.hydrosoft_forces.fill_(4.0)
    sensor._solver._reset_mask.zero_()
    before_output = sensor._output.clone()
    before_force = sensor._solver.hydrosoft_forces.clone()
    sensor.reset(torch.empty(0, dtype=torch.int64, device="cuda:0"))
    sensor.reset(torch.zeros(64, dtype=torch.bool, device="cuda:0"))
    torch.testing.assert_close(sensor._output, before_output)
    torch.testing.assert_close(sensor._solver.hydrosoft_forces, before_force)
    assert not torch.any(sensor._solver._reset_mask)
    sensor.close()


def test_non_default_torch_stream_orders_pose_advance_and_reset():
    sensor = create_tactile_sensor(
        model="hydroshear",
        num_envs=1,
        config=hydro_config(GEL, surface_points=16),
    )
    sensor.materialize(make_scene())
    sensor.enable_stage_timing(True)
    positions_np, quaternions_np = full_body_poses(
        1, np.asarray([[[0.0, 0.0, 0.001]]], dtype=np.float32)
    )
    stream = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(stream):
        positions = torch.as_tensor(positions_np, device="cuda:0")
        quaternions = torch.as_tensor(quaternions_np, device="cuda:0")
        sensor.update_pose(positions, quaternions)
        sensor.advance()
        output = sensor.read()["displacement"]
        output_pointer = output.data_ptr()
        pressed = output.clone()
        sensor.reset(torch.tensor([0], dtype=torch.int64, device="cuda:0"))
        cleared = output.clone()
    stream.synchronize()

    assert output.data_ptr() == output_pointer
    assert torch.count_nonzero(pressed) > 0
    assert torch.count_nonzero(cleared) == 0
    assert set(sensor.stage_timings_ms) == {"pose_input", "transform", "depth", "sdf", "hydrosoft"}
    sensor.close()


@pytest.mark.parametrize(
    "indices", [[-1], [64], [3, 3]], ids=["negative", "upper_bound", "duplicate"]
)
def test_cuda_invalid_integer_selectors_fail_asynchronously_in_isolated_process(indices):
    script = """
import torch
from unitac.hydroshear.sensor import _device_environment_rows
selector = torch.tensor(INDICES, dtype=torch.int64, device='cuda:0')
try:
    _device_environment_rows(selector, 64)
    torch.cuda.synchronize('cuda:0')
except (RuntimeError, IndexError):
    pass
else:
    raise SystemExit('invalid selector was accepted')
""".replace("INDICES", repr(indices))
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_selector_on_another_cuda_device_is_rejected():
    sensor = create_tactile_sensor(
        model="hydroshear",
        num_envs=4,
        config={**hydro_config(GEL, surface_points=1), "device": "cuda:0"},
    )
    sensor.materialize(make_scene())
    with pytest.raises(ValueError, match="expected cuda:0"):
        sensor.reset(torch.tensor([0], device="cuda:1"))
    with cp.cuda.Device(1):
        other_gpu_indices = cp.asarray([0], dtype=cp.int64)
    with pytest.raises(ValueError, match="expected cuda:0"):
        sensor.reset(other_gpu_indices)
    sensor.close()


def _selector(kind: str, selected: np.ndarray, num_envs: int):
    if kind == "mask":
        mask = np.zeros(num_envs, dtype=np.bool_)
        mask[selected] = True
        return torch.as_tensor(mask, device="cuda:0")
    if kind == "indices":
        return torch.as_tensor(selected, dtype=torch.int64, device="cuda:0")
    return cp.asarray(selected, dtype=cp.int64)
