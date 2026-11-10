"""HydroShear-specific calibrated RGB and diagnostic image rendering."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .depth import (
    CAMERA_DIST_M,
    CAMERA_TAN_H,
    CAMERA_TAN_V,
    DEPTH_H,
    DEPTH_W,
)
from .sensor import HydroShearVisualizationSnapshot

VisualizationMode = Literal["rgb", "depth"]
CalibrationProfile = Literal["taxim", "gelsight_mini"]

_TAXIM_FILES = ("polycalib.npz", "dataPack.npz")
_GELSIGHT_MINI_FILES = (
    "polycalib.npz",
    "mmint_gs_mini_2D9C-N227_07072025_bg_R_no_dots.png",
    "mmint_bg.jpg",
)
_GEL_WIDTH_MM = 20.75
_GEL_HEIGHT_MM = 25.25
_MARKER_DISPLACEMENT_SCALE_MM = 0.05
_MARKER_ARROW_MIN_PIXELS = 1.5
_MARKER_ARROW_TIP_LENGTH = 0.16


def _calibration_normals(height_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate calibrated image-plane slopes from the rendered height field."""
    top = height_pixels[:-2, 1:-1]
    bottom = height_pixels[2:, 1:-1]
    left = height_pixels[1:-1, :-2]
    right = height_pixels[1:-1, 2:]
    dzdx = (bottom - top) * 0.5
    dzdy = (right - left) * 0.5
    magnitude_raw = np.sqrt(dzdx * dzdx + dzdy * dzdy)
    magnitude = np.arctan(magnitude_raw)
    direction = np.zeros_like(magnitude)
    nonzero = magnitude_raw != 0.0
    direction[nonzero] = np.arctan2(dzdx[nonzero], dzdy[nonzero])
    return (
        np.pad(magnitude, ((1, 1), (1, 1)), mode="symmetric"),
        np.pad(direction, ((1, 1), (1, 1)), mode="symmetric"),
    )


def _load_lut(path: Any) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        bins = int(np.asarray(archive["bins"]).item())
        gradients = tuple(
            np.asarray(archive[name], dtype=np.float32) for name in ("grad_r", "grad_g", "grad_b")
        )
    expected_shape = (bins, bins, 6)
    if bins != 125 or any(gradient.shape != expected_shape for gradient in gradients):
        raise ValueError(f"calibration gradients must have shape {(125, 125, 6)}")
    if any(not np.isfinite(gradient).all() for gradient in gradients):
        raise ValueError("calibration gradients must be finite")
    return bins, gradients[0], gradients[1], gradients[2]


def _read_rgb(cv2: Any, path: Any) -> np.ndarray:
    buffer = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # type: ignore[attr-defined]
    if image is None:
        raise ValueError(f"calibration image could not be decoded: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]


def _resource_directory(name: str) -> Any:
    return resources.files("unitac.hydroshear").joinpath("assets").joinpath(name)


def _depth_in_native_pixels(
    depth_mm: np.ndarray,
    *,
    width: int,
    height: int,
    pixel_scale_mm: float,
    cv2: Any,
) -> np.ndarray:
    """Project HydroShear's camera raster onto a calibrated sensor pixel grid.

    HydroShear stores a pinhole-camera raster, while tactile calibrations
    describe a physical image plane. This maps target pixel
    centers to the source camera rays using the sampled surface depth to recover
    each ray's intersection. Millimetres are preserved; out-of-view pixels
    remain unindented instead of stretching HydroShear's camera image to fill
    the calibration.
    """
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.shape != (DEPTH_H, DEPTH_W):
        raise ValueError(f"depth image must have shape {(DEPTH_H, DEPTH_W)}, got {depth.shape}")

    x_mm = (np.arange(width, dtype=np.float32) - (width - 1) / 2) * pixel_scale_mm
    y_mm = (np.arange(height, dtype=np.float32) - (height - 1) / 2) * pixel_scale_mm
    camera_distance_mm = CAMERA_DIST_M * 1000.0
    target_x_mm = np.broadcast_to(x_mm[None, :], (height, width))
    target_y_mm = np.broadcast_to(y_mm[:, None], (height, width))
    surface_depth = np.zeros((height, width), dtype=np.float32)
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    for _ in range(2):
        camera_distance = np.maximum(camera_distance_mm + surface_depth, 1.0)
        map_x = (
            (target_x_mm / (camera_distance * CAMERA_TAN_H) + 1.0) * DEPTH_W * 0.5 - 0.5
        ).astype(np.float32)
        map_y = (
            (target_y_mm / (camera_distance * CAMERA_TAN_V) + 1.0) * DEPTH_H * 0.5 - 0.5
        ).astype(np.float32)
        surface_depth = cv2.remap(
            depth,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,  # type: ignore[attr-defined]
            borderMode=cv2.BORDER_CONSTANT,  # type: ignore[attr-defined]
            borderValue=0.0,
        )
    return cv2.remap(  # type: ignore[attr-defined]
        depth,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,  # type: ignore[attr-defined]
        borderMode=cv2.BORDER_CONSTANT,  # type: ignore[attr-defined]
        borderValue=0.0,
    )


def _gelsight_mini_depth_native_mm(
    depth_mm: np.ndarray,
    *,
    width: int,
    height: int,
    pixel_scale_mm: float,
    gel_width_mm: float,
    gel_height_mm: float,
    cv2: Any,
) -> np.ndarray:
    """Crop the Mini's calibrated active area and resize it to camera pixels."""
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.shape != (DEPTH_H, DEPTH_W):
        raise ValueError(f"depth image must have shape {(DEPTH_H, DEPTH_W)}, got {depth.shape}")
    active_width_mm = width * pixel_scale_mm
    active_height_mm = height * pixel_scale_mm
    source_h, source_w = depth.shape
    crop_w = max(2, int(round(source_w * active_width_mm / gel_width_mm)))
    crop_h = max(2, int(round(source_h * active_height_mm / gel_height_mm)))
    x0 = max(0, (source_w - crop_w) // 2)
    y0 = max(0, (source_h - crop_h) // 2)
    active_depth = cv2.resize(
        depth[y0 : y0 + crop_h, x0 : x0 + crop_w],
        (width, height),
        interpolation=cv2.INTER_LINEAR,  # type: ignore[attr-defined]
    )
    return np.asarray(active_depth, dtype=np.float32)


def _smooth_resampled_height(
    height_pixels: np.ndarray, *, pixel_scale_mm: float, cv2: Any
) -> np.ndarray:
    """Filter the upsampled depth raster at half a HydroShear source pixel."""
    camera_distance_mm = CAMERA_DIST_M * 1000.0
    sigma_x = camera_distance_mm * CAMERA_TAN_H / DEPTH_W / pixel_scale_mm * 0.5
    sigma_y = camera_distance_mm * CAMERA_TAN_V / DEPTH_H / pixel_scale_mm * 0.5
    return cv2.GaussianBlur(
        height_pixels.astype(np.float32),
        (0, 0),
        sigmaX=sigma_x,
        sigmaY=sigma_y,
        borderType=cv2.BORDER_REFLECT,
    )


def _marker_pixel_grid(
    marker_positions_mm: np.ndarray,
    *,
    width: int,
    height: int,
    pixel_scale_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(marker_positions_mm, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("marker positions must have shape (rows, columns, 3)")
    x = positions[..., 0] / pixel_scale_mm + (width - 1) * 0.5
    y = positions[..., 1] / pixel_scale_mm + (height - 1) * 0.5
    return x, y


def _draw_synthetic_markers(
    image: np.ndarray,
    displacement_mm: np.ndarray,
    marker_positions_mm: np.ndarray,
    *,
    pixel_scale_mm: float,
    displacement_gain: float,
    cv2: Any,
) -> np.ndarray:
    """Overlay a labeled synthetic dot grid at measured HydroShear positions."""
    displacement = np.asarray(displacement_mm, dtype=np.float32)
    positions = np.asarray(marker_positions_mm, dtype=np.float32)
    if (
        displacement.shape != positions.shape
        or displacement.ndim != 3
        or displacement.shape[-1] != 3
    ):
        raise ValueError(
            "marker positions and displacement must have matching (rows, cols, 3) grids"
        )
    if not np.isfinite(displacement_gain) or displacement_gain < 0:
        raise ValueError("displacement_gain must be finite and nonnegative")
    image = image.copy()
    rest_x, rest_y = _marker_pixel_grid(
        positions, width=image.shape[1], height=image.shape[0], pixel_scale_mm=pixel_scale_mm
    )
    current_x = (
        rest_x
        + displacement[..., 0]
        * _MARKER_DISPLACEMENT_SCALE_MM
        * displacement_gain
        / pixel_scale_mm
    )
    current_y = (
        rest_y
        + displacement[..., 1]
        * _MARKER_DISPLACEMENT_SCALE_MM
        * displacement_gain
        / pixel_scale_mm
    )
    for row in range(displacement.shape[0]):
        for col in range(displacement.shape[1]):
            start = (int(round(rest_x[row, col])), int(round(rest_y[row, col])))
            end = (int(round(current_x[row, col])), int(round(current_y[row, col])))
            if 0 <= end[0] < image.shape[1] and 0 <= end[1] < image.shape[0]:
                cv2.line(image, start, end, (0, 220, 255), 1, cv2.LINE_AA)  # type: ignore[attr-defined]
                cv2.circle(image, end, 4, (12, 12, 12), -1, cv2.LINE_AA)  # type: ignore[attr-defined]
                cv2.circle(image, end, 3, (255, 230, 40), 1, cv2.LINE_AA)  # type: ignore[attr-defined]
    return image


def _draw_displacement_vectors(
    image: np.ndarray,
    displacement_mm: np.ndarray,
    marker_positions_mm: np.ndarray,
    depth_mm: np.ndarray,
    *,
    active_width_mm: float,
    active_height_mm: float,
    display_gain: float,
    cv2: Any,
) -> np.ndarray:
    displacement = np.asarray(displacement_mm, dtype=np.float32)
    positions = np.asarray(marker_positions_mm, dtype=np.float32)
    if (
        displacement.shape != positions.shape
        or displacement.ndim != 3
        or displacement.shape[-1] != 3
    ):
        raise ValueError(
            "marker positions and displacement must have matching (rows, cols, 3) grids"
        )
    if not np.isfinite(display_gain) or display_gain < 0:
        raise ValueError("display_gain must be finite and nonnegative")
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    if (
        not np.isfinite(active_width_mm)
        or not np.isfinite(active_height_mm)
        or active_width_mm <= 0
        or active_height_mm <= 0
    ):
        raise ValueError("active image dimensions must be positive and finite")
    image = image.copy()
    image_h, image_w = image.shape[:2]
    depth_h, depth_w = depth.shape
    display_displacement = displacement[..., :2] * _MARKER_DISPLACEMENT_SCALE_MM
    markers_display = positions[..., :2] + display_displacement
    marker_u = (markers_display[..., 0] / active_width_mm + 0.5) * depth_w
    marker_v = (markers_display[..., 1] / active_height_mm + 0.5) * depth_h
    pu = np.clip(np.rint(marker_u), 0, depth_w - 1).astype(np.intp)
    pv = np.clip(np.rint(marker_v), 0, depth_h - 1).astype(np.intp)
    indent_markers = depth[pv, pu]
    start_x = (markers_display[..., 0] / active_width_mm + 0.5) * image_w
    start_y = (markers_display[..., 1] / active_height_mm + 0.5) * image_h
    delta_x = (
        display_displacement[..., 0] * display_gain * image_w / active_width_mm
    )
    delta_y = (
        display_displacement[..., 1] * display_gain * image_h / active_height_mm
    )
    arrow_length = np.hypot(delta_x, delta_y)
    keep = (indent_markers < -0.05) & (arrow_length >= _MARKER_ARROW_MIN_PIXELS)
    start = np.rint(np.stack((start_x[keep], start_y[keep]), axis=-1)).astype(np.intp)
    end = np.rint(
        np.stack((start_x[keep] + delta_x[keep], start_y[keep] + delta_y[keep]), axis=-1)
    ).astype(np.intp)
    for (x0, y0), (x1, y1) in zip(start, end, strict=True):
        cv2.arrowedLine(
            image,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            (30, 30, 30),
            2,
            cv2.LINE_AA,
            tipLength=_MARKER_ARROW_TIP_LENGTH,
        )
        cv2.arrowedLine(
            image,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
            tipLength=_MARKER_ARROW_TIP_LENGTH,
        )
    return image


@dataclass(frozen=True, slots=True)
class HydroShearVisualizationFrame:
    """A rendered view and the untouched physical values for one sensor pad."""

    env_id: int
    generation: int
    pad_index: int
    pad_name: str
    mode: VisualizationMode
    calibration_label: str
    marker_label: str
    image_rgb: np.ndarray
    raw_displacement_mm: np.ndarray
    display_displacement_mm: np.ndarray
    depth_mm: np.ndarray
    peak_displacement_mm: float
    peak_depth_mm: float


class TaximRenderer:
    """CPU renderer using the original 640x480 Taxim calibration."""

    width = 640
    height = 480
    pixel_scale_mm = 0.0295
    calibration_label = "Taxim-calibrated GelSight RGB"
    marker_label = "Synthetic dots + HydroShear arrows"

    def __init__(self) -> None:
        import cv2

        self._cv2 = cv2
        asset_root = _resource_directory("taxim")
        missing = [name for name in _TAXIM_FILES if not asset_root.joinpath(name).is_file()]
        if missing:
            raise FileNotFoundError(f"bundled Taxim assets are missing: {', '.join(missing)}")
        self.bins, self.grad_r, self.grad_g, self.grad_b = _load_lut(
            asset_root.joinpath("polycalib.npz")
        )
        data_pack = BytesIO(asset_root.joinpath("dataPack.npz").read_bytes())
        with np.load(data_pack, allow_pickle=False) as pack:
            f0 = np.asarray(pack["f0"], dtype=np.uint8)
        if f0.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Taxim f0 must have shape {(self.height, self.width, 3)}, got {f0.shape}"
            )
        self.background = self._process_initial_frame(f0)
        yy, xx = np.indices((self.height, self.width), dtype=np.float32)
        self._polynomial_basis = np.stack(
            (xx * xx, yy * yy, xx * yy, xx, yy, np.ones_like(xx)), axis=-1
        )

    def _process_initial_frame(self, f0: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        smooth = cv2.GaussianBlur(f0, (401, 401), 50.0, borderType=cv2.BORDER_REFLECT)
        mix = (smooth.astype(np.float32) - f0.astype(np.float32)).mean(axis=2) < 5.0
        background = f0.copy()
        blended = (0.15 * smooth.astype(np.float32) + 0.85 * f0).astype(np.uint8)
        background[mix] = blended[mix]
        return background

    def depth_native_mm(self, depth_mm: np.ndarray) -> np.ndarray:
        return _depth_in_native_pixels(
            depth_mm,
            width=self.width,
            height=self.height,
            pixel_scale_mm=self.pixel_scale_mm,
            cv2=self._cv2,
        )

    def _optical_rgb(self, height_pixels: np.ndarray) -> np.ndarray:
        magnitude, direction = _calibration_normals(height_pixels)
        x_bin = 0.5 * np.pi / (self.bins - 1)
        y_bin = 2.0 * np.pi / (self.bins - 1)
        ix = np.clip(np.floor(magnitude / x_bin).astype(np.intp), 0, self.bins - 1)
        iy = np.clip(np.floor((direction + np.pi) / y_bin).astype(np.intp), 0, self.bins - 1)
        channels = []
        for gradients in (self.grad_r, self.grad_g, self.grad_b):
            parameters = gradients[ix, iy]
            channels.append(np.einsum("...k,...k->...", self._polynomial_basis, parameters))
        delta = np.stack(channels, axis=-1)
        return np.clip(self.background.astype(np.float32) + delta, 0.0, 255.0).astype(np.uint8)

    def render_rgb(
        self,
        depth_mm: np.ndarray,
        displacement_mm: np.ndarray,
        marker_positions_mm: np.ndarray,
        *,
        displacement_gain: float = 1.0,
    ) -> np.ndarray:
        depth_native = self.depth_native_mm(depth_mm)
        height_pixels = _smooth_resampled_height(
            -depth_native / self.pixel_scale_mm,
            pixel_scale_mm=self.pixel_scale_mm,
            cv2=self._cv2,
        )
        image = self._optical_rgb(height_pixels)
        return _draw_synthetic_markers(
            image,
            displacement_mm,
            marker_positions_mm,
            pixel_scale_mm=self.pixel_scale_mm,
            displacement_gain=displacement_gain,
            cv2=self._cv2,
        )

    def render_depth(self, depth_mm: np.ndarray) -> np.ndarray:
        depth_native = self.depth_native_mm(depth_mm)
        depth_min = min(float(depth_native.min()), -0.1)
        normalized = np.clip(-depth_native / max(abs(depth_min), 0.1), 0.0, 1.0)
        colored = self._cv2.applyColorMap(
            np.rint(normalized * 255).astype(np.uint8), self._cv2.COLORMAP_INFERNO
        )
        return self._cv2.cvtColor(colored, self._cv2.COLOR_BGR2RGB)


class GelSightMiniRenderer:
    """Renderer using bundled or caller-supplied GelSight Mini calibration data."""

    width = 240
    height = 320
    pixel_scale_mm = 0.065
    gel_width_mm = _GEL_WIDTH_MM
    gel_height_mm = _GEL_HEIGHT_MM
    calibration_label = "GelSight Mini-calibrated RGB"
    marker_label = "Camera dots + HydroShear arrows"

    def __init__(self, calibration_dir: Path | str | None = None) -> None:
        import cv2

        self._cv2 = cv2
        external_calibration = calibration_dir is not None
        root = (
            Path(calibration_dir).expanduser().resolve()
            if external_calibration
            else _resource_directory("gelsight_mini")
        )
        paths = tuple(root.joinpath(name) for name in _GELSIGHT_MINI_FILES)
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"GelSight Mini calibration at {root} is missing: {', '.join(missing)}"
            )
        if external_calibration:
            self.calibration_label = "External GelSight Mini-calibrated RGB"
        self.bins, self.grad_r, self.grad_g, self.grad_b = _load_lut(paths[0])
        self.background = _read_rgb(cv2, paths[1])
        self.marker_background = _read_rgb(cv2, paths[2])
        expected = (self.height, self.width, 3)
        if self.background.shape != expected or self.marker_background.shape != expected:
            raise ValueError(f"GelSight Mini background images must have shape {expected}")
        self._marker_delta = self.marker_background.astype(np.float32) - self.background.astype(
            np.float32
        )
        yy, xx = np.indices((self.height, self.width), dtype=np.float32)
        self._pixel_x = xx
        self._pixel_y = yy
        self._polynomial_basis = np.stack(
            (xx * xx, yy * yy, xx * yy, xx, yy, np.ones_like(xx)), axis=-1
        )

    def depth_native_mm(self, depth_mm: np.ndarray) -> np.ndarray:
        return _gelsight_mini_depth_native_mm(
            depth_mm,
            width=self.width,
            height=self.height,
            pixel_scale_mm=self.pixel_scale_mm,
            gel_width_mm=self.gel_width_mm,
            gel_height_mm=self.gel_height_mm,
            cv2=self._cv2,
        )

    def _marker_displacement_field(
        self, displacement_mm: np.ndarray, marker_positions_mm: np.ndarray
    ) -> np.ndarray:
        displacement = np.asarray(displacement_mm, dtype=np.float32)
        positions = np.asarray(marker_positions_mm, dtype=np.float32)
        if (
            displacement.shape != positions.shape
            or displacement.ndim != 3
            or displacement.shape[-1] != 3
        ):
            raise ValueError(
                "marker positions and displacement must have matching (rows, cols, 3) grids"
            )
        field = self._cv2.resize(
            displacement[..., :2],
            (self.width, self.height),
            interpolation=self._cv2.INTER_LINEAR,
        )
        return np.asarray(field, dtype=np.float32) * _MARKER_DISPLACEMENT_SCALE_MM

    def render_rgb(
        self,
        depth_mm: np.ndarray,
        displacement_mm: np.ndarray,
        marker_positions_mm: np.ndarray,
        *,
        displacement_gain: float = 1.0,
    ) -> np.ndarray:
        depth_native = self.depth_native_mm(depth_mm)
        height_pixels = -depth_native
        height_pixels[np.abs(height_pixels) < 1e-6] = 0.0
        height_pixels = self._cv2.GaussianBlur(
            (height_pixels * 1000.0 / self.pixel_scale_mm).astype(np.float32),
            (5, 5),
            0,
        )
        magnitude, direction = _calibration_normals(height_pixels)
        ix = np.clip(
            np.floor(magnitude / (0.5 * np.pi / (self.bins - 1))).astype(np.intp),
            0,
            self.bins - 1,
        )
        iy = np.clip(
            np.floor((direction + np.pi) / (2.0 * np.pi / (self.bins - 1))).astype(np.intp),
            0,
            self.bins - 1,
        )
        channels = []
        for gradients in (self.grad_r, self.grad_g, self.grad_b):
            channels.append(np.einsum("...k,...k->...", self._polynomial_basis, gradients[ix, iy]))
        image = np.clip(
            self.background.astype(np.float32) + np.stack(channels, axis=-1), 0.0, 255.0
        ).astype(np.uint8)
        field = self._marker_displacement_field(displacement_mm, marker_positions_mm)
        map_x = self._pixel_x - (
            field[..., 0] * displacement_gain * self.width / self.gel_width_mm
        )
        map_y = self._pixel_y - (
            field[..., 1] * displacement_gain * self.height / self.gel_height_mm
        )
        warped = self._cv2.remap(
            self._marker_delta,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=self._cv2.INTER_LINEAR,
            borderMode=self._cv2.BORDER_CONSTANT,
        )
        return np.clip(image.astype(np.float32) + warped, 0.0, 255.0).astype(np.uint8)

    def render_depth(self, depth_mm: np.ndarray) -> np.ndarray:
        depth_native = self.depth_native_mm(depth_mm)
        depth_min = min(float(depth_native.min()), -0.1)
        normalized = np.clip(-depth_native / max(abs(depth_min), 0.1), 0.0, 1.0)
        colored = self._cv2.applyColorMap(
            np.rint(normalized * 255).astype(np.uint8), self._cv2.COLORMAP_INFERNO
        )
        return self._cv2.cvtColor(colored, self._cv2.COLOR_BGR2RGB)


class HydroShearVisualizer:
    """Explicit CPU diagnostic consumer for the latest HydroShear snapshot."""

    def __init__(
        self,
        *,
        mode: VisualizationMode = "rgb",
        profile: CalibrationProfile = "gelsight_mini",
        calibration_dir: Path | str | None = None,
        pad: int | str = 0,
        display_gain: float = 4.0,
    ) -> None:
        if mode not in ("rgb", "depth"):
            raise ValueError("mode must be 'rgb' or 'depth'")
        if not np.isfinite(display_gain) or display_gain < 0:
            raise ValueError("display_gain must be finite and nonnegative")
        if profile == "taxim":
            if calibration_dir is not None:
                raise ValueError("calibration_dir is only used with profile='gelsight_mini'")
            self.renderer = TaximRenderer()
        elif profile == "gelsight_mini":
            self.renderer = GelSightMiniRenderer(calibration_dir)
        else:
            raise ValueError("profile must be 'taxim' or 'gelsight_mini'")
        self.mode = mode
        self.profile = profile
        self.pad = pad
        self.display_gain = float(display_gain)
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
        if selected_mode not in ("rgb", "depth"):
            raise ValueError("mode must be 'rgb' or 'depth'")
        snapshot = sensor.read_visualization_snapshot(env_id)  # type: ignore[attr-defined]
        pad_index = self._resolve_pad(snapshot, self.pad if pad is None else pad)
        raw = np.asarray(snapshot.displacement_mm[pad_index], dtype=np.float32)
        display = raw * _MARKER_DISPLACEMENT_SCALE_MM
        depth = np.asarray(snapshot.depth_mm[pad_index], dtype=np.float32)
        marker_positions = np.asarray(snapshot.marker_positions_mm, dtype=np.float32)
        if selected_mode == "rgb":
            image = self.renderer.render_rgb(depth, raw, marker_positions, displacement_gain=1.0)
            image = _draw_displacement_vectors(
                image,
                raw,
                marker_positions,
                depth,
                active_width_mm=self.renderer.width * self.renderer.pixel_scale_mm,
                active_height_mm=self.renderer.height * self.renderer.pixel_scale_mm,
                display_gain=self.display_gain,
                cv2=self.renderer._cv2,
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
            calibration_label=self.renderer.calibration_label,
            marker_label=self.renderer.marker_label,
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
        if mode not in ("rgb", "depth"):
            raise ValueError("mode must be 'rgb' or 'depth'")
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
