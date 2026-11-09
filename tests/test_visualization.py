from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")


def _new_visualizer(**kwargs):
    from unitac.hydroshear.visualization import HydroShearVisualizer

    return HydroShearVisualizer(**kwargs)


def _calibration_dir(root: Path) -> tuple[Path, np.ndarray]:
    directory = root / "calibration"
    directory.mkdir()
    background = np.full((320, 240, 3), (80, 120, 150), dtype=np.uint8)
    markers = background.copy()
    for row in range(9):
        for col in range(7):
            center = (20 + col * 33, 20 + row * 35)
            cv2.circle(markers, center, 5, (18, 20, 22), -1)
    cv2.imwrite(
        str(directory / "mmint_gs_mini_2D9C-N227_07072025_bg_R_no_dots.png"), background[..., ::-1]
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


def test_calibrated_rgb_restores_dot_background_and_changes_under_contact(tmp_path: Path) -> None:
    calibration_dir, marker_background = _calibration_dir(tmp_path)
    displacement = np.zeros((9, 7, 3), dtype=np.float32)
    source = _SnapshotSource(_snapshot(np.zeros((175, 100), dtype=np.float32), displacement))
    visualizer = _new_visualizer(calibration_dir=calibration_dir, mode="rgb")
    no_contact = visualizer.update(source)
    np.testing.assert_array_equal(no_contact.image_rgb, visualizer.renderer.marker_background)
    np.testing.assert_array_equal(no_contact.raw_displacement_mm, 0.0)

    source.snapshot = _snapshot(np.full((175, 100), -0.8, dtype=np.float32), displacement)
    contact = visualizer.update(source)
    assert np.any(contact.image_rgb != visualizer.renderer.marker_background)
    assert contact.peak_depth_mm == pytest.approx(-0.8)
    assert source.read_count == 2
    assert marker_background.shape == (320, 240, 3)
    visualizer.close()


def test_display_gain_changes_only_display_displacement(tmp_path: Path) -> None:
    calibration_dir, _ = _calibration_dir(tmp_path)
    depth = np.full((175, 100), -0.5, dtype=np.float32)
    displacement = np.zeros((9, 7, 3), dtype=np.float32)
    displacement[..., 0] = 0.1
    displacement[..., 1] = -0.05
    source = _SnapshotSource(_snapshot(depth, displacement))
    visualizer = _new_visualizer(
        calibration_dir=calibration_dir, mode="displacement", display_gain=3.0
    )
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


def test_depth_mode_selects_signed_depth_view(tmp_path: Path) -> None:
    calibration_dir, _ = _calibration_dir(tmp_path)
    depth = np.zeros((175, 100), dtype=np.float32)
    depth[60:100, 30:70] = -1.25
    source = _SnapshotSource(_snapshot(depth, np.zeros((9, 7, 3), dtype=np.float32)))
    visualizer = _new_visualizer(calibration_dir=calibration_dir, mode="depth")
    frame = visualizer.update(source, pad="left_gel")
    assert frame.mode == "depth"
    assert frame.pad_name == "left_gel"
    assert frame.peak_depth_mm == pytest.approx(-1.25)
    assert frame.image_rgb.shape == (320, 240, 3)
    visualizer.close()
