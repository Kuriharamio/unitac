"""Shared tactile panel for the letter-press backend examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .simulation import LetterPressSimulation, observation_metrics
else:
    from simulation import LetterPressSimulation, observation_metrics


@dataclass(frozen=True)
class TactileDisplayFrame:
    panel_rgb: np.ndarray
    image_rgb: np.ndarray
    metrics: dict[str, Any]
    mode: str
    calibration_label: str
    marker_label: str


class TactileDisplay:
    """Build the same HydroShear panel for each backend's viewer."""

    def __init__(
        self,
        *,
        mode: str,
        calibration_dir: Path | None,
        display_gain: float,
    ) -> None:
        from unitac.visualization import create_visualizer

        self.visualizer = create_visualizer(
            tactile_type="hydroshear",
            mode=mode,
            calibration_dir=calibration_dir,
            display_gain=display_gain,
        )

    @property
    def mode(self) -> str:
        return self.visualizer.mode

    def set_mode(self, mode: str) -> None:
        self.visualizer.set_mode(mode)

    def update(
        self,
        simulation: LetterPressSimulation,
        *,
        phase: str,
        sim_time: float,
    ) -> TactileDisplayFrame:
        if simulation.sensor is None:
            raise RuntimeError("letter-press sensor is not initialized")
        tactile = self.visualizer.update(simulation.sensor, env_id=0)
        metrics = observation_metrics(
            tactile.raw_displacement, tactile.depth_mm, tactile.generation
        )
        panel = _tactile_panel(
            tactile.image_rgb,
            mode=tactile.mode,
            phase=phase,
            sim_time=sim_time,
            metrics=metrics,
            calibration_label=tactile.calibration_label,
            marker_label=tactile.marker_label,
        )
        return TactileDisplayFrame(
            panel_rgb=panel,
            image_rgb=tactile.image_rgb,
            metrics=metrics,
            mode=tactile.mode,
            calibration_label=tactile.calibration_label,
            marker_label=tactile.marker_label,
        )

    def close(self) -> None:
        self.visualizer.close()


def _tactile_panel(
    image_rgb: np.ndarray,
    *,
    mode: str,
    phase: str,
    sim_time: float,
    metrics: dict[str, Any],
    calibration_label: str,
    marker_label: str,
) -> np.ndarray:
    import cv2

    panel = np.full((720, 320, 3), (25, 28, 33), dtype=np.uint8)
    cv2.putText(
        panel,
        f"{calibration_label} | {mode.upper()}",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"{phase} | Simulation {sim_time:.2f} s",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (225, 230, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        marker_label,
        (12, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (80, 225, 255),
        1,
        cv2.LINE_AA,
    )
    image = cv2.resize(image_rgb, (300, 400), interpolation=cv2.INTER_AREA)
    panel[78:478, 10:310] = image
    lines = (
        f"Contact pixels: {metrics['contact_pixels']}",
        f"Peak depth: {metrics['peak_depth_mm']:.3f} mm",
        f"Peak marker: {metrics['peak_tangential_displacement_model_units']:.3f} HydroShear units",
        f"Sensor generation: {metrics['generation']}",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (16, 520 + index * 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (225, 230, 235),
            1,
            cv2.LINE_AA,
        )
    return panel
