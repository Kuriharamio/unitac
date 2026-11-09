"""Run a synthetic CUDA HydroShear press-and-slide sequence and save its field plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
from matplotlib.text import Text

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from unisim.factory import create_tactile_sensor
from unisim.ray_query import (
    RayCollisionDescription,
    RayGeomType,
    RayMeshDescription,
    RaySceneDescription,
    RaySiteFrameDescription,
)
from unisim.tactile import TactileSceneDescription

FACES = np.asarray(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int32,
)


def _box_vertices(half_extents: tuple[float, float, float]) -> np.ndarray:
    x, y, z = half_extents
    return np.asarray(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        dtype=np.float32,
    )


def _scene() -> TactileSceneDescription:
    names = ("object", "left_pad", "right_pad")
    mesh = RayMeshDescription(
        asset_id="example:box",
        points=_box_vertices((0.005, 0.005, 0.004)),
        indices=FACES.reshape(-1),
    )
    collision = RayCollisionDescription(
        scene=RaySceneDescription(
            num_bodies=3,
            geom_types=(RayGeomType.MESH,),
            geom_sizes=np.zeros((1, 3)),
            geom_local_pos=np.zeros((1, 3)),
            geom_local_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
            geom_body_ids=np.asarray([0], dtype=np.int32),
            body_names=names,
        ),
        geom_names=("object_mesh",),
        meshes=(mesh,),
        geom_mesh_ids=np.asarray([0], dtype=np.int32),
        body_parent_ids=np.asarray([-1, -1, -1], dtype=np.int32),
    )
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    frames = (
        RaySiteFrameDescription("left_gel", "left_pad", np.zeros(3), identity),
        RaySiteFrameDescription("right_gel", "right_pad", np.zeros(3), identity),
    )
    return TactileSceneDescription(collision, ("object",), frames)


def _gel_mesh() -> np.ndarray:
    vertices = _box_vertices((0.010, 0.012625, 0.003))
    vertices[:, 2] -= 0.003
    return vertices[FACES]


def _frame(x: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    body_pos = np.zeros((1, 3, 3), dtype=np.float32)
    body_pos[0, 0] = (x, 0.0, z)
    body_pos[0, 1, 0] = -0.004
    body_pos[0, 2, 0] = 0.004
    body_quat = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, 3, 1))
    return body_pos, body_quat


def run(output: Path) -> np.ndarray:
    config = {
        "gel_mesh": _gel_mesh(),
        "surface_points": 2048,
        "output_grid": {
            "rows": 9,
            "cols": 7,
            "spacing_mm": [2.145, 2.153125],
            "center_offset_mm": [-0.3575, -0.325],
            "pixel_scale_mm": 0.065,
        },
        "solver": {"use_fft_dilation": False, "use_matmul_shear": True},
        "device": "cuda:0",
    }
    sensor = create_tactile_sensor("unitac", model="hydroshear", num_envs=1, config=config)
    sensor.materialize(_scene())
    values = []
    sequence = [(0.0, 0.010), (0.0, 0.001)]
    sequence.extend((x, 0.001) for x in (0.0005, 0.0010, 0.0015, 0.0020))
    for x, z in sequence:
        body_pos, body_quat = _frame(x, z)
        sensor.update_pose(body_pos, body_quat)
        sensor.advance()
        values.append(sensor.read()["displacement"][0, 0].detach().cpu().numpy().copy())
    torch.cuda.synchronize()
    sensor.close()

    result = np.stack(values)
    if not np.isfinite(result).all():
        raise ValueError("HydroShear produced nonfinite marker displacement")
    output.parent.mkdir(parents=True, exist_ok=True)
    _plot(result, output)
    print(
        f"frames={len(result)}, markers={result.shape[1]}x{result.shape[2]}, "
        f"component_min_mm={result.min(axis=(0, 1, 2)).tolist()}, "
        f"component_max_mm={result.max(axis=(0, 1, 2)).tolist()}"
    )
    return result


def _plot(result: np.ndarray, output: Path) -> None:
    states = (result[1], result[2], result[-1])
    titles = ("Press", "Slide onset", "After 2 mm slide")
    x = (np.arange(result.shape[2]) - (result.shape[2] - 1) / 2) * 2.145 - 0.3575
    y = (np.arange(result.shape[1]) - (result.shape[1] - 1) / 2) * 2.153125 - 0.325
    xx, yy = np.meshgrid(x, y)
    normal_limit = max(float(np.abs(state[..., 2]).max()) for state in states)
    tangent_limit = max(float(np.hypot(state[..., 0], state[..., 1]).max()) for state in states)
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.2), constrained_layout=True)
    for column, (state, title) in enumerate(zip(states, titles, strict=True)):
        image = axes[0, column].pcolormesh(
            xx,
            yy,
            state[..., 2],
            cmap="RdBu_r",
            vmin=-normal_limit,
            vmax=normal_limit,
            shading="nearest",
        )
        axes[0, column].set_title(title, fontsize=9, pad=5)
        if column == 0:
            axes[0, column].set_ylabel("y (mm)")
        magnitude = np.hypot(state[..., 0], state[..., 1])
        tangent_image = axes[1, column].pcolormesh(
            xx,
            yy,
            magnitude,
            cmap="viridis",
            vmin=0.0,
            vmax=tangent_limit,
            shading="nearest",
        )
        direction_norm = np.maximum(magnitude, 1e-9)
        axes[1, column].quiver(
            xx,
            yy,
            state[..., 0] / direction_norm,
            state[..., 1] / direction_norm,
            color="#202020",
            angles="xy",
            scale_units="xy",
            scale=1.5,
            width=0.004,
        )
        axes[1, column].set_xlabel("x (mm)")
        if column == 0:
            axes[1, column].set_ylabel("y (mm)")
        else:
            axes[0, column].tick_params(labelleft=False)
            axes[1, column].tick_params(labelleft=False)
        for row in range(2):
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xticks(x[::2], labels=[f"{value:.1f}" for value in x[::2]])
            axes[row, column].set_yticks(y[::2], labels=[f"{value:.1f}" for value in y[::2]])
            axes[row, column].tick_params(labelsize=7, pad=2)
    fig.colorbar(image, ax=axes[0, :], label="dz (mm)", shrink=0.84)
    fig.colorbar(tangent_image, ax=axes[1, :], label="|dxy| (mm)", shrink=0.84)
    fig.suptitle("HydroShear marker field · sensor 0", fontsize=11)
    _audit_figure_layout(fig)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _audit_figure_layout(fig: matplotlib.figure.Figure) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for label in fig.findobj(match=Text):
        if not label.get_visible() or not label.get_text().strip():
            continue
        box = label.get_window_extent(renderer)
        if (
            box.x0 < bounds.x0 - 1
            or box.y0 < bounds.y0 - 1
            or box.x1 > bounds.x1 + 1
            or box.y1 > bounds.y1 + 1
        ):
            raise ValueError(f"figure text is clipped: {label.get_text()!r}")

    for axis in fig.axes:
        for labels in (axis.get_xticklabels(), axis.get_yticklabels()):
            visible = [label for label in labels if label.get_visible() and label.get_text()]
            boxes = [label.get_window_extent(renderer) for label in visible]
            for left, right in zip(boxes, boxes[1:], strict=False):
                if left.overlaps(right):
                    raise ValueError("figure tick labels overlap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/hydroshear_press_slide.png"))
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
