"""Run one HydroShear sample with a static MJBatch-backed MuJoCo scene."""

from __future__ import annotations

import argparse
import tempfile
from importlib import resources
from pathlib import Path

import numpy as np
import torch
from unisim.factory import create_backend, create_tactile_sensor
from unisim.scene import SceneCfg
from unisim.tactile import TactileSceneDescription

from unitac.hydroshear.config import load_triangle_mesh


def _write_scene(path: Path) -> None:
    path.write_text(
        """<mujoco model="unitac-example">
  <worldbody>
    <body name="object">
      <geom name="object_box" type="box" size="0.005 0.005 0.004"
            pos="0 0 0.001"/>
    </body>
    <body name="sensor_mount">
      <site name="gel_site"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )


def _gel_mesh() -> np.ndarray:
    asset = resources.files("unitac.hydroshear").joinpath(
        "assets", "letter_press", "gsmini_elastomer_collision_512.obj"
    )
    with resources.as_file(asset) as path:
        triangles = load_triangle_mesh(path).copy()
    return triangles + np.asarray([0.0005, 0.05975, 0.00725], dtype=np.float32)


def run() -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("HydroShear requires CUDA")

    with tempfile.TemporaryDirectory(prefix="unitac-mjbatch-") as temporary:
        scene_path = Path(temporary) / "scene.xml"
        _write_scene(scene_path)
        backend = create_backend(
            "mujoco",
            SceneCfg(model_file=str(scene_path)),
            num_envs=1,
            sim_dt=0.002,
        )
        sensor = None
        try:
            backend.materialize()
            collision = backend.get_ray_collision_description(("object", "sensor_mount"))
            scene = TactileSceneDescription(
                collision=collision,
                object_body_names=("object",),
                sensor_frames=backend.get_ray_site_frames(("gel_site",)),
                geometry_state=backend.get_ray_geometry_state(collision.scene.body_names),
            )
            sensor = create_tactile_sensor(
                "unitac",
                model="hydroshear",
                num_envs=1,
                config={"gel_mesh": _gel_mesh(), "device": "cuda:0"},
            )
            sensor.materialize(scene)

            body_count = len(collision.scene.body_names)
            body_pos = torch.zeros((1, body_count, 3), device="cuda:0")
            body_quat = torch.zeros((1, body_count, 4), device="cuda:0")
            body_quat[..., 0] = 1.0
            sensor.update_pose(body_pos, body_quat)
            sensor.advance()
            return sensor.read()["displacement"].detach().cpu().numpy().copy()
        finally:
            if sensor is not None:
                sensor.close()
            backend.close()


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    displacement = run()
    print(f"displacement shape: {displacement.shape}")


if __name__ == "__main__":
    main()
