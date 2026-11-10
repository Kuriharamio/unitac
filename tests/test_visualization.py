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
        displacement_mm=displacement[None],
        depth_mm=depth[None],
    )


def test_bundled_gelsight_mini_profile_is_native_and_uses_camera_markers() -> None:
    visualizer = _new_visualizer(mode="rgb")
    assert visualizer.profile == "gelsight_mini"
    assert visualizer.renderer.width == 240
    assert visualizer.renderer.height == 320
    assert visualizer.renderer.pixel_scale_mm == pytest.approx(0.065)
    assert visualizer.renderer.bins == 125
    assert visualizer.renderer.background.shape == (320, 240, 3)

    displacement = np.zeros((9, 7, 3), dtype=np.float32)
    source = _SnapshotSource(_snapshot(np.zeros((175, 100), dtype=np.float32), displacement))
    frame = visualizer.update(source)
    assert frame.calibration_label == "GelSight Mini-calibrated RGB"
    assert frame.marker_label == "Camera dots; motion from HydroShear displacement"
    assert frame.image_rgb.shape == (320, 240, 3)
    assert not frame.image_rgb.flags.writeable
    assert source.read_count == 1
    visualizer.close()


def test_taxim_projection_preserves_mm_per_pixel() -> None:
    renderer = _new_visualizer(profile="taxim").renderer
    source_depth = np.broadcast_to(-np.arange(100, dtype=np.float32) * 0.001, (175, 100)).copy()
    native = renderer.depth_native_mm(source_depth)
    center_y = renderer.height // 2
    center_x = renderer.width // 2
    shift = 100
    native_delta = native[center_y, center_x + shift] - native[center_y, center_x]
    target_mm = shift * renderer.pixel_scale_mm
    source_width_mm = 2 * (0.0267 * 1000 - 0.05) * np.tan(np.deg2rad(72.146676) / 2) * 240 / 320
    expected_delta = -0.001 * target_mm / source_width_mm * 100
    assert native_delta == pytest.approx(expected_delta, abs=0.002)


def test_gelsight_rgb_uses_contact_and_actual_displacement() -> None:
    displacement = np.zeros((9, 7, 3), dtype=np.float32)
    depth = np.zeros((175, 100), dtype=np.float32)
    depth[55:120, 26:70] = -0.8
    source = _SnapshotSource(_snapshot(depth, displacement))
    visualizer = _new_visualizer(mode="rgb")
    still = visualizer.update(source)

    displaced = displacement.copy()
    displaced[4, 3, 0] = 0.2
    source.snapshot = _snapshot(depth, displaced)
    moving = visualizer.update(source)
    assert np.any(still.image_rgb != moving.image_rgb)
    assert moving.peak_depth_mm == pytest.approx(-0.8)
    np.testing.assert_array_equal(moving.raw_displacement_mm, displaced)
    assert source.read_count == 2
    visualizer.close()


def test_display_gain_changes_only_display_displacement() -> None:
    depth = np.full((175, 100), -0.5, dtype=np.float32)
    displacement = np.zeros((9, 7, 3), dtype=np.float32)
    displacement[..., 0] = 0.1
    displacement[..., 1] = -0.05
    source = _SnapshotSource(_snapshot(depth, displacement))
    visualizer = _new_visualizer(mode="displacement", display_gain=3.0)
    frame = visualizer.update(source)
    np.testing.assert_array_equal(frame.raw_displacement_mm, displacement)
    np.testing.assert_array_equal(frame.display_displacement_mm, displacement * 3.0)
    np.testing.assert_array_equal(source.snapshot.displacement_mm[0], displacement)
    assert frame.image_rgb.shape == (320, 240, 3)
    visualizer.set_display_gain(0.0)
    zero_gain = visualizer.update(source)
    np.testing.assert_array_equal(zero_gain.raw_displacement_mm, displacement)
    np.testing.assert_array_equal(zero_gain.display_displacement_mm, 0.0)
    assert np.any(frame.image_rgb != zero_gain.image_rgb)
    visualizer.close()


def test_external_gelsight_profile_uses_authorized_camera_marker_assets(
    tmp_path: Path,
) -> None:
    calibration_dir, marker_background = _gelsight_mini_calibration_dir(tmp_path)
    source = _SnapshotSource(
        _snapshot(np.zeros((175, 100), dtype=np.float32), np.zeros((9, 7, 3), dtype=np.float32))
    )
    visualizer = _new_visualizer(
        profile="gelsight_mini", calibration_dir=calibration_dir, mode="rgb"
    )
    frame = visualizer.update(source)
    assert frame.calibration_label == "External GelSight Mini-calibrated RGB"
    assert frame.marker_label == "Camera dots; motion from HydroShear displacement"
    assert frame.image_rgb.shape == marker_background.shape
    visualizer.close()


def test_taxim_depth_mode_maps_signed_depth_to_taxim_grid() -> None:
    depth = np.zeros((175, 100), dtype=np.float32)
    depth[60:100, 30:70] = -1.25
    source = _SnapshotSource(_snapshot(depth, np.zeros((9, 7, 3), dtype=np.float32)))
    visualizer = _new_visualizer(profile="taxim", mode="depth")
    frame = visualizer.update(source, pad="left_gel")
    assert frame.mode == "depth"
    assert frame.pad_name == "left_gel"
    assert frame.peak_depth_mm == pytest.approx(-1.25)
    assert frame.image_rgb.shape == (480, 640, 3)
    visualizer.close()
