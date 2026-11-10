from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")


def _new_visualizer(**kwargs):
    from unitac.hydroshear.visualization import HydroShearVisualizer

    return HydroShearVisualizer(**kwargs)


def _gelsight_mini_calibration_dir(root: Path) -> tuple[Path, np.ndarray]:
    directory = root / "calibration"
    directory.mkdir()
    background = np.full((320, 240, 3), (80, 120, 150), dtype=np.uint8)
    markers = background.copy()
    for row in range(9):
        for col in range(7):
            center = (20 + col * 33, 20 + row * 35)
            cv2.circle(markers, center, 5, (18, 20, 22), -1)
    cv2.imwrite(
        str(directory / "mmint_gs_mini_2D9C-N227_07072025_bg_R_no_dots.png"),
        background[..., ::-1],
    )
    cv2.imwrite(str(directory / "mmint_bg.jpg"), markers[..., ::-1])
    grad_r = np.zeros((125, 125, 6), dtype=np.float32)
    grad_r[..., 5] = 8.0
    zeros = np.zeros_like(grad_r)
    np.savez(
        directory / "polycalib.npz",
        bins=np.asarray(125),
        grad_r=grad_r,
        grad_g=zeros,
        grad_b=zeros,
    )
    return directory, markers


class _SnapshotSource:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.read_count = 0

    def read_visualization_snapshot(self, env_id: int) -> object:
        self.read_count += 1
        assert env_id == self.snapshot.env_id
        return self.snapshot


def _snapshot(depth: np.ndarray, displacement: np.ndarray) -> object:
    from unitac.hydroshear.sensor import HydroShearVisualizationSnapshot

    rows, cols, _ = displacement.shape
    x = (np.arange(cols, dtype=np.float32) - (cols - 1) / 2) * 2.0
    y = (np.arange(rows, dtype=np.float32) - (rows - 1) / 2) * 2.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    markers = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
    return HydroShearVisualizationSnapshot(
        env_id=0,
        generation=4,
        site_names=("left_gel",),
        marker_positions_mm=markers,
        displacement=displacement[None],
        depth_mm=depth[None],
    )


def test_bundled_gelsight_mini_is_default_and_uses_camera_markers() -> None:
    visualizer = _new_visualizer(mode="rgb")
    assert visualizer.display_gain == 4.0
    assert visualizer.renderer.width == 240
    assert visualizer.renderer.height == 320
    assert visualizer.renderer.pixel_scale_mm == pytest.approx(0.065)
    assert visualizer.renderer.bins == 125
    assert visualizer.renderer.background.shape == (320, 240, 3)

    displacement = np.zeros((9, 7, 3), dtype=np.float32)
    source = _SnapshotSource(_snapshot(np.zeros((175, 100), dtype=np.float32), displacement))
    frame = visualizer.update(source)
    assert frame.calibration_label == "GelSight Mini-calibrated RGB"
    assert frame.marker_label == "Camera dots + HydroShear arrows"
    assert frame.image_rgb.shape == (320, 240, 3)
    assert not frame.image_rgb.flags.writeable
    assert source.read_count == 1
    visualizer.close()


def test_gelsight_depth_maps_millimetres_to_calibrated_pixels_and_slope() -> None:
    from unitac.hydroshear.visualization import (
        _calibration_normals,
        _height_pixels_from_depth,
        _normal_lut_indices,
    )

    flat_mm = np.full((11, 11), -0.065, dtype=np.float32)
    np.testing.assert_allclose(_height_pixels_from_depth(flat_mm, 0.065), 1.0)

    zero_mm = np.zeros_like(flat_mm)
    zero_height = _height_pixels_from_depth(zero_mm, 0.065)
    np.testing.assert_array_equal(zero_height, 0.0)
    zero_magnitude, _ = _calibration_normals(zero_height)
    np.testing.assert_array_equal(zero_magnitude, 0.0)

    x_pixels = np.broadcast_to(np.arange(11, dtype=np.float32), (11, 11))
    plane_mm = -x_pixels * 0.065
    height_pixels = _height_pixels_from_depth(plane_mm, 0.065)
    magnitude, direction = _calibration_normals(height_pixels)
    ix, iy = _normal_lut_indices(magnitude, direction, 125)
    assert magnitude[5, 5] == pytest.approx(np.arctan(1.0), abs=1e-6)
    assert direction[5, 5] == pytest.approx(0.0, abs=1e-6)
    assert ix[5, 5] == 62
    assert iy[5, 5] == 62
    assert int(ix.max()) < 124
    assert int(iy.max()) < 124


def test_gelsight_rgb_uses_native_depth_scale_and_non_saturated_lut(monkeypatch) -> None:
    from unitac.hydroshear.visualization import (
        _calibration_normals,
        _height_pixels_from_depth,
        _normal_lut_indices,
    )

    renderer = _new_visualizer().renderer
    flat_depth = np.full((175, 100), -0.065, dtype=np.float32)
    native_flat = renderer.depth_native_mm(flat_depth)
    np.testing.assert_allclose(native_flat, -0.065, atol=1e-7)
    np.testing.assert_allclose(
        _height_pixels_from_depth(native_flat, renderer.pixel_scale_mm), 1.0, atol=2e-6
    )

    source = np.zeros((175, 100), dtype=np.float32)
    displacement = np.zeros((9, 7, 3), dtype=np.float32)
    marker_positions = np.zeros_like(displacement)
    images = []
    for gradient_mm_per_pixel in (0.065, 0.13):
        x = np.broadcast_to(
            np.arange(renderer.width, dtype=np.float32),
            (renderer.height, renderer.width),
        )
        native = -gradient_mm_per_pixel * x
        monkeypatch.setattr(renderer, "depth_native_mm", lambda _depth, value=native: value)
        height = _height_pixels_from_depth(native, renderer.pixel_scale_mm)
        height = cv2.GaussianBlur(height.astype(np.float32), (5, 5), 0)
        magnitude, direction = _calibration_normals(height)
        ix, iy = _normal_lut_indices(magnitude, direction, renderer.bins)
        assert int(ix.max()) < renderer.bins - 1
        assert int(iy.max()) < renderer.bins - 1
        images.append(renderer.render_rgb(source, displacement, marker_positions))
    assert not np.array_equal(images[0][160, 120], images[1][160, 120])


def test_gelsight_rgb_uses_contact_and_actual_displacement() -> None:
    displacement = np.zeros((9, 7, 3), dtype=np.float32)
    depth = np.zeros((175, 100), dtype=np.float32)
    depth[55:120, 26:70] = -0.8
    source = _SnapshotSource(_snapshot(depth, displacement))
    visualizer = _new_visualizer(mode="rgb")
    still = visualizer.update(source)

    displaced = displacement.copy()
    displaced[4, 3, 0] = 1.0
    source.snapshot = _snapshot(depth, displaced)
    moving = visualizer.update(source)
    assert np.any(still.image_rgb != moving.image_rgb)
    assert moving.peak_depth_mm == pytest.approx(-0.8)
    np.testing.assert_array_equal(moving.raw_displacement, displaced)
    assert source.read_count == 2
    visualizer.close()


def test_display_gain_changes_only_arrow_scale_on_rgb_layer() -> None:
    depth = np.full((175, 100), -0.5, dtype=np.float32)
    displacement = np.zeros((9, 7, 3), dtype=np.float32)
    displacement[..., 0] = 0.8
    displacement[..., 1] = -0.4
    source = _SnapshotSource(_snapshot(depth, displacement))
    visualizer = _new_visualizer(mode="rgb", display_gain=4.0)
    frame = visualizer.update(source)
    assert frame.mode == "rgb"
    np.testing.assert_array_equal(frame.raw_displacement, displacement)
    np.testing.assert_array_equal(frame.display_displacement, displacement * 0.05)
    np.testing.assert_array_equal(source.snapshot.displacement[0], displacement)
    assert frame.image_rgb.shape == (320, 240, 3)
    visualizer.set_display_gain(0.0)
    zero_gain = visualizer.update(source)
    np.testing.assert_array_equal(zero_gain.raw_displacement, displacement)
    np.testing.assert_array_equal(zero_gain.display_displacement, displacement * 0.05)
    assert np.any(frame.image_rgb != zero_gain.image_rgb)
    visualizer.close()


def test_rgb_arrow_directions_follow_sensor_local_xy() -> None:
    from unitac.hydroshear.visualization import _draw_displacement_vectors

    background = np.zeros((100, 100, 3), dtype=np.uint8)
    marker_positions = np.zeros((1, 1, 3), dtype=np.float32)
    displacement_x = np.asarray([[[1.3, 0.0, 0.0]]], dtype=np.float32)
    displacement_y = np.asarray([[[0.0, 1.3, 0.0]]], dtype=np.float32)
    visualizer = _new_visualizer()
    renderer = visualizer.renderer

    right = _draw_displacement_vectors(
        background,
        displacement_x,
        marker_positions,
        np.full((175, 100), -0.5, dtype=np.float32),
        active_width_mm=6.5,
        active_height_mm=6.5,
        display_gain=4.0,
        cv2=renderer._cv2,
    )
    down = _draw_displacement_vectors(
        background,
        displacement_y,
        marker_positions,
        np.full((175, 100), -0.5, dtype=np.float32),
        active_width_mm=6.5,
        active_height_mm=6.5,
        display_gain=4.0,
        cv2=renderer._cv2,
    )
    green_right = (right[..., 1] > 200) & (right[..., 0] < 50) & (right[..., 2] < 50)
    green_down = (down[..., 1] > 200) & (down[..., 0] < 50) & (down[..., 2] < 50)
    assert np.any(green_right[48:52, 51:57])
    assert not np.any(green_right[51:57, 48:52])
    assert np.any(green_down[51:57, 48:52])
    assert not np.any(green_down[48:52, 51:57])
    visualizer.close()


def test_arrows_are_hidden_without_contact_or_below_minimum_length() -> None:
    depth = np.zeros((175, 100), dtype=np.float32)
    displacement = np.zeros((9, 7, 3), dtype=np.float32)
    displacement[..., 0] = 0.1
    source = _SnapshotSource(_snapshot(depth, displacement))
    visualizer = _new_visualizer(mode="rgb", display_gain=4.0)
    with_arrows_enabled = visualizer.update(source).image_rgb
    visualizer.set_display_gain(0.0)
    without_arrows = visualizer.update(source).image_rgb
    np.testing.assert_array_equal(with_arrows_enabled, without_arrows)
    visualizer.close()


def test_external_gelsight_profile_uses_authorized_camera_marker_assets(
    tmp_path: Path,
) -> None:
    calibration_dir, marker_background = _gelsight_mini_calibration_dir(tmp_path)
    source = _SnapshotSource(
        _snapshot(np.zeros((175, 100), dtype=np.float32), np.zeros((9, 7, 3), dtype=np.float32))
    )
    visualizer = _new_visualizer(
        calibration_dir=calibration_dir, mode="rgb"
    )
    frame = visualizer.update(source)
    assert frame.calibration_label == "External GelSight Mini-calibrated RGB"
    assert frame.marker_label == "Camera dots + HydroShear arrows"
    assert frame.image_rgb.shape == marker_background.shape
    visualizer.close()
