"""GelSight Mini calibrated image formation and HydroShear display helpers.

The renderer implements the TACSL/Taxim calibrated gradient lookup used by
HydroShear's original GelSight Mini example. Calibration data is resolved
from packaged resources or an explicit external directory; it is never
silently synthesized from displacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np

from .sensor import HydroShearVisualizationSnapshot

VisualizationMode = Literal["rgb", "displacement", "depth"]

_CALIBRATION_FILES = (
    "polycalib.npz",
    "mmint_gs_mini_2D9C-N227_07072025_bg_R_no_dots.png",
    "mmint_bg.jpg",
)


def _taxim_normals(height_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the squashed gradient magnitude and direction used by Taxim."""
    dzdx, dzdy = np.gradient(height_map)
    dzdx = dzdx[1:-1, 1:-1]
    dzdy = dzdy[1:-1, 1:-1]
    magnitude_raw = np.sqrt(dzdx * dzdx + dzdy * dzdy)
    magnitude = np.arctan(magnitude_raw)
    direction = np.arctan2(dzdx, dzdy)
    direction[magnitude_raw == 0] = 0
    return (
        np.pad(magnitude, ((1, 1), (1, 1)), mode="symmetric"),
        np.pad(direction, ((1, 1), (1, 1)), mode="symmetric"),
    )


@dataclass(frozen=True, slots=True)
class HydroShearVisualizationFrame:
    """A rendered view and the untouched physical values for one sensor pad."""

    env_id: int
    generation: int
    pad_index: int
    pad_name: str
    mode: VisualizationMode
    image_rgb: np.ndarray
    raw_displacement_mm: np.ndarray
    display_displacement_mm: np.ndarray
    depth_mm: np.ndarray
    peak_displacement_mm: float
    peak_depth_mm: float


class GelSightMiniRenderer:
    """CPU renderer for calibrated GelSight Mini RGB and diagnostics."""

    width = 240
    height = 320
    pixel_scale_mm = 0.065
    active_width_mm = width * pixel_scale_mm
    active_height_mm = height * pixel_scale_mm

    def __init__(self, calibration_dir: Path | str | None = None) -> None:
        import cv2

        self._cv2 = cv2
        if calibration_dir is not None:
            root = Path(calibration_dir).expanduser().resolve()
            paths = tuple(root / name for name in _CALIBRATION_FILES)
            missing = [path.name for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"calibration directory {root} is missing required files: {', '.join(missing)}"
                )
            self._load_files(*paths)
        else:
            package_assets = (
                resources.files("unitac.hydroshear").joinpath("assets").joinpath("gelsight_mini")
            )
            missing = [
                name for name in _CALIBRATION_FILES if not package_assets.joinpath(name).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "licensed GelSight Mini calibration assets are not packaged. "
                    "Pass calibration_dir pointing to authorized TACSL files; "
                    f"missing: {', '.join(missing)}"
                )
            with resources.as_file(package_assets) as root:
                self._load_files(*(root / name for name in _CALIBRATION_FILES))

    def _load_files(self, calibration_path: Path, background_path: Path, marker_path: Path) -> None:
        cv2 = self._cv2
        background_bgr = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        marker_bgr = cv2.imread(str(marker_path), cv2.IMREAD_COLOR)
        if background_bgr is None or marker_bgr is None:
            raise ValueError("GelSight Mini background images could not be decoded")
        self.background = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
        self.marker_background = cv2.cvtColor(marker_bgr, cv2.COLOR_BGR2RGB)
        expected = (self.height, self.width, 3)
        if self.background.shape != expected:
            self.background = cv2.resize(self.background, (self.width, self.height))
        if self.marker_background.shape != expected:
            self.marker_background = cv2.resize(self.marker_background, (self.width, self.height))
        with np.load(calibration_path) as archive:
            self.bins = int(np.asarray(archive["bins"]).item())
            self.grad_r = np.asarray(archive["grad_r"], dtype=np.float32)
            self.grad_g = np.asarray(archive["grad_g"], dtype=np.float32)
            self.grad_b = np.asarray(archive["grad_b"], dtype=np.float32)
        if self.bins != 125:
            raise ValueError(f"unsupported GelSight Mini calibration bins: {self.bins}")
        expected_lut = (self.bins, self.bins, 6)
        if any(array.shape != expected_lut for array in (self.grad_r, self.grad_g, self.grad_b)):
            raise ValueError(f"calibration gradient planes must have shape {expected_lut}")
        yy, xx = np.indices((self.height, self.width), dtype=np.float32)
        self._polynomial_basis = np.stack(
            (xx * xx, yy * yy, xx * yy, xx, yy, np.ones_like(xx)), axis=-1
        )
        self._x_bin = 0.5 * np.pi / (self.bins - 1)
        self._y_bin = 2.0 * np.pi / (self.bins - 1)
        self._marker_delta = self.marker_background.astype(np.float32) - self.background.astype(
            np.float32
        )

    def render_rgb(
        self,
        depth_mm: np.ndarray,
        displacement_mm: np.ndarray,
        marker_positions_mm: np.ndarray,
        *,
        displacement_gain: float = 1.0,
    ) -> np.ndarray:
        """Render calibrated RGB and warp the printed dots using marker motion."""
        depth = np.asarray(depth_mm, dtype=np.float32)
        displacement = np.asarray(displacement_mm, dtype=np.float32)
        markers = np.asarray(marker_positions_mm, dtype=np.float32)
        if depth.shape != (175, 100):
            raise ValueError(f"depth image must have shape (175, 100), got {depth.shape}")
        if displacement.ndim != 3 or displacement.shape[-1] != 3:
            raise ValueError("displacement must have shape (rows, columns, 3)")
        if markers.shape != displacement.shape:
            raise ValueError("marker positions and displacement must have matching grids")
        if not np.isfinite(displacement_gain) or displacement_gain < 0:
            raise ValueError("displacement_gain must be finite and nonnegative")

        cv2 = self._cv2
        # depth is a signed intrusion image (zero outside contact, negative in
        # contact); the optical model consumes positive surface height.
        source_h, source_w = depth.shape
        crop_w = max(2, int(round(source_w * self.active_width_mm / 20.75)))
        crop_h = max(2, int(round(source_h * self.active_height_mm / 25.25)))
        x0, y0 = (source_w - crop_w) // 2, (source_h - crop_h) // 2
        optical_depth = cv2.resize(
            depth[y0 : y0 + crop_h, x0 : x0 + crop_w],
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )
        height = -optical_depth
        contact = height > 1e-6
        if not np.any(contact):
            return self.marker_background.copy()
        height_mm = cv2.GaussianBlur(
            (height * 1000.0 / self.pixel_scale_mm).astype(np.float32), (5, 5), 0
        )
        magnitude, direction = _taxim_normals(height_mm)
        ix = np.clip(np.floor(magnitude / self._x_bin).astype(np.intp), 0, self.bins - 1)
        iy = np.clip(
            np.floor((direction + np.pi) / self._y_bin).astype(np.intp),
            0,
            self.bins - 1,
        )
        parameters = (self.grad_r[ix, iy], self.grad_g[ix, iy], self.grad_b[ix, iy])
        gradients = np.stack(
            [np.sum(self._polynomial_basis * channel, axis=-1) for channel in parameters],
            axis=-1,
        )
        rgb = np.clip(gradients + self.background.astype(np.float32), 0.0, 255.0).astype(np.uint8)
        # Taxim's calibrated RGB is unchanged where there is no indentation.
        rgb[~contact] = self.background[~contact]

        field = cv2.resize(
            (displacement[..., :2] * displacement_gain).astype(np.float32),
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )
        # Marker coordinates are in sensor-local mm; image columns follow +x,
        # and image rows follow +y. The field is interpolated over the optical
        # active area, so a physical mm maps to 1 / pixel_scale_mm pixels.
        yy, xx = np.indices((self.height, self.width), dtype=np.float32)
        map_x = xx - field[..., 0] / self.pixel_scale_mm
        map_y = yy - field[..., 1] / self.pixel_scale_mm
        warped_delta = cv2.remap(
            self._marker_delta,
            np.asarray(map_x, dtype=np.float32),
            np.asarray(map_y, dtype=np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return np.clip(rgb.astype(np.float32) + warped_delta, 0.0, 255.0).astype(np.uint8)

    def render_displacement(
        self,
        displacement_mm: np.ndarray,
        marker_positions_mm: np.ndarray,
        *,
        display_gain: float = 1.0,
    ) -> np.ndarray:
        """Render marker vectors; display_gain never modifies sensor output."""
        cv2 = self._cv2
        displacement = np.asarray(displacement_mm, dtype=np.float32)
        markers = np.asarray(marker_positions_mm, dtype=np.float32)
        if (
            displacement.shape != markers.shape
            or displacement.ndim != 3
            or displacement.shape[-1] != 3
        ):
            raise ValueError(
                "marker positions and displacement must have matching (rows, cols, 3) grids"
            )
        if not np.isfinite(display_gain) or display_gain < 0:
            raise ValueError("display_gain must be finite and nonnegative")
        image = self.marker_background.copy()
        px = (markers[..., 0] / self.active_width_mm + 0.5) * (self.width - 1)
        py = (markers[..., 1] / self.active_height_mm + 0.5) * (self.height - 1)
        vector = displacement[..., :2] * display_gain / self.pixel_scale_mm
        for row in range(displacement.shape[0]):
            for col in range(displacement.shape[1]):
                start = (int(round(px[row, col])), int(round(py[row, col])))
                end = (
                    int(round(px[row, col] + vector[row, col, 0])),
                    int(round(py[row, col] + vector[row, col, 1])),
                )
                cv2.arrowedLine(image, start, end, (25, 25, 25), 2, cv2.LINE_AA, tipLength=0.25)
                cv2.arrowedLine(image, start, end, (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.25)
                cv2.circle(image, start, 2, (255, 215, 0), -1, cv2.LINE_AA)
        return image

    def render_depth(self, depth_mm: np.ndarray) -> np.ndarray:
        """Render signed contact depth (negative mm) with an inferno scale."""
        depth = np.asarray(depth_mm, dtype=np.float32)
        if depth.shape != (175, 100):
            raise ValueError(f"depth image must have shape (175, 100), got {depth.shape}")
        depth_min = min(float(depth.min()), -0.1)
        normalized = np.clip(-depth / max(abs(depth_min), 0.1), 0.0, 1.0)
        colored = self._cv2.applyColorMap(
            np.rint(normalized * 255).astype(np.uint8), self._cv2.COLORMAP_INFERNO
        )
        rgb = self._cv2.cvtColor(colored, self._cv2.COLOR_BGR2RGB)
        return self._cv2.resize(
            rgb, (self.width, self.height), interpolation=self._cv2.INTER_NEAREST
        )


class HydroShearVisualizer:
    """Explicit CPU diagnostic consumer for the latest HydroShear snapshot."""

    def __init__(
        self,
        *,
        mode: VisualizationMode = "rgb",
        calibration_dir: Path | str | None = None,
        pad: int | str = 0,
        display_gain: float = 4.0,
    ) -> None:
        if mode not in ("rgb", "displacement", "depth"):
            raise ValueError("mode must be 'rgb', 'displacement', or 'depth'")
        if not np.isfinite(display_gain) or display_gain < 0:
            raise ValueError("display_gain must be finite and nonnegative")
        self.mode = mode
        self.pad = pad
        self.display_gain = float(display_gain)
        self.renderer = GelSightMiniRenderer(calibration_dir)
        self.last_frame: HydroShearVisualizationFrame | None = None
        self._closed = False

    def update(
        self,
        sensor: object,
        *,
        env_id: int = 0,
        pad: int | str | None = None,
        mode: VisualizationMode | None = None,
    ) -> HydroShearVisualizationFrame:
        if self._closed:
            raise RuntimeError("visualizer is closed")
        selected_mode = self.mode if mode is None else mode
        if selected_mode not in ("rgb", "displacement", "depth"):
            raise ValueError("mode must be 'rgb', 'displacement', or 'depth'")
        snapshot = sensor.read_visualization_snapshot(env_id)  # type: ignore[attr-defined]
        pad_index = self._resolve_pad(snapshot, self.pad if pad is None else pad)
        raw = np.asarray(snapshot.displacement_mm[pad_index], dtype=np.float32)
        display = raw * self.display_gain
        depth = np.asarray(snapshot.depth_mm[pad_index], dtype=np.float32)
        marker_positions = np.asarray(snapshot.marker_positions_mm, dtype=np.float32)
        if selected_mode == "rgb":
            image = self.renderer.render_rgb(depth, raw, marker_positions, displacement_gain=1.0)
        elif selected_mode == "displacement":
            image = self.renderer.render_displacement(
                raw, marker_positions, display_gain=self.display_gain
            )
        else:
            image = self.renderer.render_depth(depth)
        raw_copy = raw.copy()
        display_copy = display.copy()
        depth_copy = depth.copy()
        for array in (image, raw_copy, display_copy, depth_copy):
            array.setflags(write=False)
        frame = HydroShearVisualizationFrame(
            env_id=snapshot.env_id,
            generation=snapshot.generation,
            pad_index=pad_index,
            pad_name=snapshot.site_names[pad_index],
            mode=selected_mode,
            image_rgb=image,
            raw_displacement_mm=raw_copy,
            display_displacement_mm=display_copy,
            depth_mm=depth_copy,
            peak_displacement_mm=float(np.linalg.norm(raw[..., :2], axis=-1).max()),
            peak_depth_mm=float(depth.min()),
        )
        self.last_frame = frame
        return frame

    def set_mode(self, mode: VisualizationMode) -> None:
        if mode not in ("rgb", "displacement", "depth"):
            raise ValueError("mode must be 'rgb', 'displacement', or 'depth'")
        self.mode = mode

    def set_display_gain(self, gain: float) -> None:
        if not np.isfinite(gain) or gain < 0:
            raise ValueError("display_gain must be finite and nonnegative")
        self.display_gain = float(gain)

    @staticmethod
    def _resolve_pad(snapshot: HydroShearVisualizationSnapshot, pad: int | str) -> int:
        if isinstance(pad, str):
            try:
                return snapshot.site_names.index(pad)
            except ValueError as error:
                raise KeyError(
                    f"unknown tactile site {pad!r}; available: {snapshot.site_names}"
                ) from error
        if isinstance(pad, bool) or not isinstance(pad, (int, np.integer)):
            raise TypeError("pad must be an integer index or site name")
        index = int(pad)
        if not 0 <= index < len(snapshot.site_names):
            raise IndexError(f"pad must be in [0, {len(snapshot.site_names)})")
        return index

    def close(self) -> None:
        self.last_frame = None
        self._closed = True
