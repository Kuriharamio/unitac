from __future__ import annotations

import pytest
import torch

from unitac.hydroshear.solver import HydroSoftSensor


def _force_frame(sensor: HydroSoftSensor) -> torch.Tensor:
    sensor._reset_mask.zero_()
    sensor.hydrosoft_forces.zero_()
    sensor.prev_sdf.fill_(1.0)
    sensor.prev_grasped_obj_pts_elastomer.zero_()
    current = torch.tensor([[[2.0, 1.0, 3.0]], [[2.0, 1.0, 3.0]]], dtype=torch.float32)
    sdf = torch.full((2, 1), -1.0)
    return sensor.step_hydrosoft_forces(current, sdf, reverse_z=True)


def test_hydrosoft_applies_scalar_and_per_environment_force_coefficients():
    sensor = HydroSoftSensor(
        device="cpu",
        mu=(0.5, 1.0),
        E=(2.0, 1.0),
        K=(4.0, 2.0),
        A=(3.0, 1.0),
    )
    sensor.initialize(2, 1)

    forces = _force_frame(sensor)

    assert forces.shape == (2, 1, 3)
    torch.testing.assert_close(forces[0, 0, 2], torch.tensor(-9.0))
    torch.testing.assert_close(forces[0, 0, :2], torch.tensor([-4.024922, -2.012461]))
    torch.testing.assert_close(forces[1, 0, 2], torch.tensor(-1.5))
    torch.testing.assert_close(forces[1, 0, :2], torch.tensor([-1.3416407, -0.67082036]))


def test_hydrosoft_shear_and_dilation_parameters_change_marker_output():
    marker_points = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]])
    heights = torch.tensor([[0.001, 0.0]])
    object_points = torch.tensor([[[0.001, 0.0, 0.001]]])
    forces = torch.tensor([[[0.0001, 0.0002, 0.0003]]])

    default = HydroSoftSensor(device="cpu", use_matmul_shear=True)
    default.initialize(1, 1)
    default_dilation = default.get_dilation_displacement(marker_points, heights)
    default_shear = default.get_marker_displacement_components_from_force(
        marker_points, heights, object_points, forces
    )[1]

    tuned = HydroSoftSensor(
        device="cpu",
        lambda_d=1.0,
        lambda_s=1.0,
        dilate_scale=2.0,
        shear_scale=2.0,
        use_matmul_shear=True,
    )
    tuned.initialize(1, 1)
    tuned_dilation = tuned.get_dilation_displacement(marker_points, heights)
    tuned_shear = tuned.get_marker_displacement_components_from_force(
        marker_points, heights, object_points, forces
    )[1]

    assert not torch.allclose(default_dilation, tuned_dilation)
    assert not torch.allclose(default_shear, tuned_shear)


def test_hydrosoft_rejects_unsupported_axis_and_wrong_environment_coefficient_shape():
    for axis in (0, 1, 3, True):
        with pytest.raises(ValueError, match="only normal_axis=2"):
            HydroSoftSensor(device="cpu", normal_axis=axis)

    sensor = HydroSoftSensor(device="cpu", mu=(0.2, 0.4, 0.6))
    with pytest.raises(ValueError, match="scalar or have 2 values"):
        sensor.initialize(2, 1)


def test_hydrosoft_rejects_ineffective_and_non_finite_direct_parameters():
    with pytest.raises(ValueError, match="only normal_axis=2"):
        HydroSoftSensor(device="cpu", normal_axis=2.0)
    with pytest.raises(TypeError, match="use_matmul_shear must be bool"):
        HydroSoftSensor(device="cpu", use_matmul_shear=1)
    with pytest.raises(ValueError, match="positive integer"):
        HydroSoftSensor(device="cpu", fft_dilation_min_pts=256.5)
    with pytest.raises(ValueError, match="finite and non-negative"):
        HydroSoftSensor(device="cpu", mu=float("nan")).initialize(1, 1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        HydroSoftSensor(device="cpu", E=-1.0).initialize(1, 1)
    with pytest.raises(ValueError, match="one-dimensional"):
        HydroSoftSensor(device="cpu", K=[[1.0]]).initialize(1, 1)


def test_hydrosoft_reset_rejects_bad_cpu_selectors_and_preserves_other_history():
    sensor = HydroSoftSensor(device="cpu")
    sensor.initialize(3, 2)
    sensor._reset_mask.zero_()
    sensor.hydrosoft_forces.fill_(1.0)
    sensor.prev_sdf.fill_(2.0)
    sensor.prev_grasped_obj_pts_elastomer.fill_(3.0)

    with pytest.raises(IndexError, match="out of range"):
        sensor.reset([-1])
    with pytest.raises(IndexError, match="distinct"):
        sensor.reset([1, 1])
    with pytest.raises(TypeError, match="integers or booleans"):
        sensor.reset([1.5])
    with pytest.raises(ValueError, match="one-dimensional"):
        sensor.reset([[0]])

    sensor.reset([1])

    torch.testing.assert_close(sensor.hydrosoft_forces[0], torch.ones((2, 3)))
    torch.testing.assert_close(sensor.hydrosoft_forces[1], torch.zeros((2, 3)))
    torch.testing.assert_close(sensor.hydrosoft_forces[2], torch.ones((2, 3)))
    torch.testing.assert_close(sensor.prev_sdf, torch.full((3, 2), 2.0))
    torch.testing.assert_close(sensor.prev_grasped_obj_pts_elastomer, torch.full((3, 2, 3), 3.0))
    torch.testing.assert_close(sensor._reset_mask, torch.tensor([False, True, False]))


def test_hydrosoft_empty_reset_selector_is_a_noop():
    sensor = HydroSoftSensor(device="cpu")
    sensor.initialize(2, 1)
    sensor._reset_mask.zero_()
    sensor.hydrosoft_forces.fill_(1.0)

    sensor.reset([])

    torch.testing.assert_close(sensor.hydrosoft_forces, torch.ones((2, 1, 3)))
    torch.testing.assert_close(sensor._reset_mask, torch.zeros(2, dtype=torch.bool))
