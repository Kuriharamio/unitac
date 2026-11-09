from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from unitac import create_tactile_sensor

from .helpers import box_triangles, full_body_poses, hydro_config, make_scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="HydroShear requires CUDA")
FIXTURE = Path(__file__).parent / "fixtures" / "hydroshear_reference.npz"
GEL = box_triangles((0.010, 0.012625, 0.003))
GEL[:, :, 2] -= 0.003


def _pose(x: float = 0.0, z: float = 0.010, yaw: float = 0.0):
    quat = np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)
    return np.asarray([x, 0.0, z], dtype=np.float32), quat


def _single_object_frame(sensor, object_count: int, poses):
    positions = np.zeros((1, object_count, 3), dtype=np.float32)
    quaternions = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, object_count, 1))
    for index, (position, quaternion) in enumerate(poses):
        positions[0, index] = position
        quaternions[0, index] = quaternion
    body_pos, body_quat = full_body_poses(1, positions, quaternions)
    sensor.update_pose(body_pos, body_quat)
    sensor.advance()


def _run_reference_case(name: str, *, keep_all_envs: bool = False):
    ref = np.load(FIXTURE)
    object_count = 2 if name == "multi_object" else 1
    scene = make_scene(tuple(f"object{index}" for index in range(object_count)))
    sensor = create_tactile_sensor(
        model="hydroshear",
        num_envs=2 if name == "partial_reset" else 1,
        config=hydro_config(GEL),
    )
    sensor.materialize(scene)
    if name == "no_contact":
        poses = [_pose(z=0.010), _pose(z=0.010)]
        for pose in poses:
            _single_object_frame(sensor, 1, [pose])
    elif name in ("press", "translation", "rotation", "sustained_contact", "sliding"):
        poses = [_pose(z=0.010), _pose(z=0.001)]
        if name == "translation":
            poses.append(_pose(x=0.001, z=0.001))
        elif name == "rotation":
            poses.append(_pose(z=0.001, yaw=0.25))
        elif name == "sliding":
            poses.extend(_pose(x=x, z=0.001) for x in (0.0005, 0.001, 0.002))
        elif name == "sustained_contact":
            poses.extend((_pose(z=0.001), _pose(z=0.001)))
        for pose in poses:
            _single_object_frame(sensor, 1, [pose])
    elif name == "multi_object":
        positions = np.asarray([[[0.0, 0.0, 0.001], [0.008, 0.0, 0.001]]], dtype=np.float32)
        quaternions = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, 2, 1))
        body_pos, body_quat = full_body_poses(1, positions, quaternions)
        sensor.update_pose(body_pos, body_quat)
        sensor.advance()
    elif name == "partial_reset":
        for frame_index, z in enumerate((0.010, 0.001, 0.001)):
            positions = np.zeros((2, 1, 3), dtype=np.float32)
            positions[:, :, 2] = z
            body_pos, body_quat = full_body_poses(2, positions)
            if frame_index == 2:
                sensor.reset([1])
            sensor.update_pose(body_pos, body_quat)
            sensor.advance()
    else:
        raise AssertionError(name)
    torch.cuda.synchronize()
    output = sensor.read()["displacement"].detach().cpu().numpy()
    if name == "partial_reset" and not keep_all_envs:
        output = output[1:2]
    sensor.close()
    return output, ref[name]


@pytest.mark.parametrize(
    "name",
    [
        "no_contact",
        "press",
        "translation",
        "rotation",
        "sliding",
        "sustained_contact",
        "multi_object",
        "partial_reset",
    ],
)
def test_hydroshear_matches_fixed_reference(name):
    actual, expected = _run_reference_case(name)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)


def test_pose_update_does_not_advance_and_read_reuses_sensor_buffer():
    sensor = create_tactile_sensor(model="hydroshear", num_envs=1, config=hydro_config(GEL))
    sensor.materialize(make_scene())
    pose = np.asarray(
        [[[0.0, 0.0, 0.001], [-0.004, 0.0, 0.0], [0.004, 0.0, 0.0]]], dtype=np.float32
    )
    quat = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, 3, 1))
    sensor.update_pose(pose, quat)
    with pytest.raises(RuntimeError, match="advance must complete"):
        sensor.read()
    sensor.advance()
    first_ptr = sensor.read()["displacement"].data_ptr()
    sensor.update_pose(pose, quat)
    assert sensor.read()["displacement"].data_ptr() == first_ptr
    assert sensor.output_specs[0].buffer_owner == "sensor"
    assert sensor.output_specs[0].units == "mm"
    sensor.close()


def test_numpy_cpu_torch_and_cuda_pose_profiles_match():
    pose = np.asarray(
        [[[0.0, 0.0, 0.001], [-0.004, 0.0, 0.0], [0.004, 0.0, 0.0]]],
        dtype=np.float32,
    )
    quaternion = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, 3, 1))
    outputs = []
    for profile in ("numpy", "torch_cpu", "torch_cuda"):
        sensor = create_tactile_sensor(model="hydroshear", num_envs=1, config=hydro_config(GEL))
        sensor.materialize(make_scene())
        if profile == "numpy":
            position_input, quaternion_input = pose, quaternion
        elif profile == "torch_cpu":
            position_input = torch.from_numpy(pose)
            quaternion_input = torch.from_numpy(quaternion)
        else:
            position_input = torch.from_numpy(pose).cuda()
            quaternion_input = torch.from_numpy(quaternion).cuda()
        sensor.update_pose(position_input, quaternion_input)
        sensor.advance()
        outputs.append(sensor.read()["displacement"].detach().cpu().numpy().copy())
        sensor.close()

    np.testing.assert_allclose(outputs[0], outputs[1], rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(outputs[0], outputs[2], rtol=2e-6, atol=2e-5)


def test_partial_reset_preserves_unselected_environment_history():
    actual, expected = _run_reference_case("partial_reset", keep_all_envs=True)
    np.testing.assert_allclose(actual[1:2], expected, rtol=2e-6, atol=2e-5)
    assert np.linalg.norm(actual[0] - actual[1]) > 1e-4


def test_rebuild_requires_fresh_pose_update_and_close_is_idempotent():
    sensor = create_tactile_sensor(model="hydroshear", num_envs=1, config=hydro_config(GEL))
    sensor.materialize(make_scene())
    sensor.advance()
    sensor.rebuild(make_scene())
    with pytest.raises(RuntimeError, match="advance must complete"):
        sensor.read()
    with pytest.raises(RuntimeError, match="update_pose"):
        sensor.advance()
    pose = np.asarray(
        [[[0.0, 0.0, 0.001], [-0.004, 0.0, 0.0], [0.004, 0.0, 0.0]]],
        dtype=np.float32,
    )
    quaternion = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, 3, 1))
    sensor.update_pose(pose, quaternion)
    sensor.advance()
    sensor.close()
    sensor.close()
    with pytest.raises(RuntimeError, match="closed"):
        sensor.read()
