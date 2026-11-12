"""Run one HydroShear sample from the packaged letter press scene."""

from __future__ import annotations

import argparse
from importlib import resources

import numpy as np
import torch
from unisim import TensorIOSpec
from unisim.factory import create_backend, create_tactile_sensor
from unisim.scene import SceneCfg
from unisim.tactile import TactileSceneDescription

from unitac.hydroshear.config import load_triangle_mesh


_BODY_NAMES = ("pad", "indenter")
_SITE_NAMES = ("gel_surface",)
_GEL_OFFSET = np.asarray([0.0005, 0.05975, 0.00725], dtype=np.float32)


def run() -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("HydroShear requires CUDA")

    asset_root = resources.files("unitac.hydroshear").joinpath("assets", "letter_press")
    scene_asset = asset_root.joinpath("scene.xml")
    with resources.as_file(scene_asset) as scene_path:
        backend = create_backend(
            "mujoco",
            SceneCfg(model_file=str(scene_path)),
            num_envs=1,
            sim_dt=0.002,
            base_name="indenter",
            body_state_required=True,
        )
        sensor = None
        pose_plan = None
        try:
            backend.materialize()
            collision = backend.get_ray_collision_description(body_names=_BODY_NAMES)
            scene = TactileSceneDescription(
                collision=collision,
                object_body_names=("indenter",),
                object_geom_names=("letter_geom",),
                sensor_frames=backend.get_ray_site_frames(_SITE_NAMES),
                geometry_state=backend.get_ray_geometry_state(collision.scene.body_names),
            )
            gel_mesh_path = scene_path.parent / "gsmini_elastomer_collision_512.obj"
            gel_mesh = load_triangle_mesh(gel_mesh_path).copy() + _GEL_OFFSET
            sensor = create_tactile_sensor(
                tactile_type="hydroshear",
                num_envs=1,
                config={"gel_mesh": gel_mesh, "device": "cuda:0"},
            )
            sensor.materialize(scene)
            pose_plan = backend.bind_host_bridge_io(
                TensorIOSpec(
                    body_names=_BODY_NAMES,
                    body_fields=("pos_w", "quat_w"),
                    device="cpu",
                )
            )
            pose_plan.read_state_sensors()
            poses = pose_plan.get_body_pose_views()
            if poses.pos_w is None or poses.quat_w is None:
                raise RuntimeError("UniSim did not provide both body position and quaternion views")
            sensor.update_pose(poses.pos_w, poses.quat_w)
            sensor.advance()
            return sensor.read()["displacement"].detach().cpu().numpy().copy()
        finally:
            if pose_plan is not None:
                pose_plan.close()
            if sensor is not None:
                sensor.close()
            backend.close()


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    displacement = run()
    print(f"displacement shape: {displacement.shape}")


if __name__ == "__main__":
    main()
