"""Run HydroShear from public UniSim MuJoCo or MJWarp APIs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from unisim import TensorIOSpec
from unisim.factory import create_backend, create_tactile_sensor
from unisim.scene import SceneCfg
from unisim.tactile import TactileSceneDescription

from unitac.geometry import body_meshes
from unitac.hydroshear.config import load_triangle_mesh

BODY_NAMES = ("object", "left_pad", "right_pad", "static_fixture")
SITE_NAMES = ("left_gel", "right_gel")
MESH_VERTICES = np.asarray(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)
MESH_FACES = np.asarray(
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


def _model_xml(path: Path) -> None:
    vertices = " ".join(f"{value:g}" for value in MESH_VERTICES.reshape(-1))
    faces = " ".join(str(value) for value in MESH_FACES.reshape(-1))
    path.write_text(
        f"""<mujoco model="unitac-real-backend">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <asset>
    <mesh name="scaled_cube" vertex="{vertices}" face="{faces}"
          scale="0.005 0.005 0.004"/>
  </asset>
  <worldbody>
    <body name="object">
      <freejoint name="object_free"/>
      <geom name="object_mesh_geom" type="mesh" mesh="scaled_cube"
            pos="0.001 0.002 0.003"/>
      <geom name="excluded_aux_geom" type="sphere" size="0.001"
            pos="0 0.02 0"/>
    </body>
    <body name="left_pad" mocap="true" pos="-0.004 0 0">
      <site name="left_gel" pos="0.001 0 0"
            quat="0.9238795325 0 0 0.3826834324"/>
    </body>
    <body name="right_pad" mocap="true" pos="0.004 0 0">
      <site name="right_gel" pos="-0.001 0 0"
            quat="0.9659258263 0 0 -0.2588190451"/>
    </body>
    <body name="static_fixture" pos="0.03 0.03 0.02">
      <geom name="fixture_mesh_geom" type="mesh" mesh="scaled_cube"/>
    </body>
  </worldbody>
</mujoco>""",
        encoding="utf-8",
    )


def _gel_mesh() -> np.ndarray:
    path = Path(__file__).parents[1] / "tests/fixtures/gsmini_elastomer_collision_512.obj"
    triangles = load_triangle_mesh(path).copy()
    triangles += np.asarray([0.0005, 0.05975, 0.00725], dtype=np.float32)
    return triangles


def _hydro_config() -> dict[str, Any]:
    return {
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


def run(backend_type: str = "mujoco") -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("the real backend HydroShear example requires CUDA")
    if backend_type not in ("mujoco", "mjwarp"):
        raise ValueError("backend_type must be 'mujoco' or 'mjwarp'")
    with tempfile.TemporaryDirectory(prefix="unitac-mujoco-") as temporary:
        xml_path = Path(temporary) / "scene.xml"
        _model_xml(xml_path)
        backend_kwargs: dict[str, Any] = {
            "base_name": "object",
            "body_state_required": True,
            "add_body_sensors": True,
        }
        if backend_type == "mujoco":
            backend_kwargs["tracked_body_names"] = BODY_NAMES
        backend = create_backend(
            backend_type,
            SceneCfg(model_file=str(xml_path)),
            num_envs=1,
            sim_dt=0.002,
            **backend_kwargs,
        )
        sensor = None
        pose_plan = None
        try:
            backend.materialize()
            collision = backend.get_ray_collision_description(body_names=BODY_NAMES)
            geometry_state = backend.get_ray_geometry_state(body_names=BODY_NAMES)
            site_frames = backend.get_ray_site_frames(SITE_NAMES)
            if collision.scene.body_names != BODY_NAMES:
                raise AssertionError(f"unexpected backend body order: {collision.scene.body_names}")
            if tuple(frame.site_name for frame in site_frames) != SITE_NAMES:
                raise AssertionError("backend site frame order changed")
            if tuple(frame.body_name for frame in site_frames) != ("left_pad", "right_pad"):
                raise AssertionError("backend site frames are attached to unexpected bodies")
            if "excluded_aux_geom" not in collision.geom_names:
                raise AssertionError(
                    f"named geometry selection fixture is absent: {collision.geom_names}"
                )

            converted_object = body_meshes(
                collision,
                ("object",),
                geom_names=("object_mesh_geom",),
                geometry_state=geometry_state,
            )[0]
            expected_min = np.asarray([-0.004, -0.003, -0.001], dtype=np.float32)
            expected_max = np.asarray([0.006, 0.007, 0.007], dtype=np.float32)
            np.testing.assert_allclose(converted_object.min(axis=(0, 1)), expected_min, atol=2e-6)
            np.testing.assert_allclose(converted_object.max(axis=(0, 1)), expected_max, atol=2e-6)

            scene = TactileSceneDescription(
                collision=collision,
                object_body_names=("object",),
                sensor_frames=site_frames,
                object_geom_names=("object_mesh_geom",),
                geometry_state=geometry_state,
            )
            sensor = create_tactile_sensor(
                "unitac", model="hydroshear", num_envs=1, config=_hydro_config()
            )
            sensor.materialize(scene)
            if backend_type == "mujoco":
                pose_plan = backend.bind_host_bridge_io(
                    TensorIOSpec(
                        body_names=BODY_NAMES,
                        body_fields=("pos_w", "quat_w"),
                        device="cpu",
                    )
                )
            pad_bindings = tuple(backend.bind_mocap_pose(name) for name in BODY_NAMES[1:3])
            mocap_poses = (
                np.asarray([[-0.004, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
                np.asarray([[0.004, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
            )
            actions = (
                ("no_contact", (0.0, 0.0, 0.020)),
                ("press", (0.0, 0.0, -0.002)),
                ("slide", (0.001, 0.0, -0.002)),
                ("retreat", (0.0, 0.0, 0.020)),
            )
            outputs: dict[str, np.ndarray] = {}
            observed_body_names: tuple[str, ...] | None = None
            state_device = "cpu" if backend_type == "mujoco" else "cuda:0"
            for action, object_position in actions:
                state = backend.get_state_views(("qpos", "qvel"), device=state_device)
                qpos = state["qpos"].clone()
                qpos[0, :3] = torch.tensor(object_position, dtype=torch.float32, device=qpos.device)
                qpos[0, 3:7] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=qpos.device
                )
                qvel = torch.zeros_like(state["qvel"])
                backend.set_state_tensor(
                    torch.tensor([0], dtype=torch.int64, device=qpos.device),
                    qpos.contiguous(),
                    qvel,
                )
                for binding, poses in zip(pad_bindings, mocap_poses, strict=True):
                    binding.write(np.asarray([0], dtype=np.int64), poses)
                backend.step_tensor(
                    torch.zeros(
                        (1, backend.num_actuators), dtype=torch.float32, device=qpos.device
                    ),
                    nsteps=1,
                )
                if backend_type == "mujoco":
                    pose_plan.read_state_sensors()
                    body_poses = pose_plan.get_body_pose_views()
                else:
                    body_poses = backend.get_tracked_body_views(BODY_NAMES, device="cuda:0")
                if observed_body_names is None:
                    observed_body_names = body_poses.body_names
                if body_poses.body_names != BODY_NAMES:
                    raise AssertionError(f"unexpected pose body order: {body_poses.body_names}")
                if not torch.allclose(body_poses.pos_w[0, 0], qpos[0, :3]):
                    raise AssertionError(
                        "public object world pose disagrees with its free-joint pose"
                    )
                sensor.update_pose(body_poses.pos_w, body_poses.quat_w)
                sensor.advance()
                outputs[action] = sensor.read()["displacement"].detach().cpu().numpy().copy()

            no_contact = outputs["no_contact"]
            press = outputs["press"]
            slide = outputs["slide"]
            no_contact_norm = float(np.linalg.norm(no_contact))
            press_norm = float(np.linalg.norm(press))
            slide_delta = slide[..., :2] - press[..., :2]
            slide_delta_norm = float(np.linalg.norm(slide_delta))
            slide_delta_mean = slide_delta.mean(axis=(0, 1, 2, 3))
            if no_contact_norm > 1e-5:
                raise AssertionError(f"no-contact action produced displacement {no_contact_norm}")
            if press_norm < 1e-4:
                raise AssertionError("press action did not produce tactile displacement")
            if slide_delta_norm < 1e-4:
                raise AssertionError("slide action did not change the tangential marker field")
            if slide_delta_mean[0] <= 0.0:
                raise AssertionError("+X object slide did not produce a +X sensor-local response")

            return {
                "backend": backend_type,
                "body_names": collision.scene.body_names,
                "pose_body_names": observed_body_names,
                "geom_names": collision.geom_names,
                "selected_object_geoms": scene.object_geom_names,
                "mesh_assets": tuple(mesh.asset_id for mesh in collision.meshes),
                "object_mesh_body_local_bounds": {
                    "min": converted_object.min(axis=(0, 1)).tolist(),
                    "max": converted_object.max(axis=(0, 1)).tolist(),
                },
                "site_frames": tuple(
                    {
                        "site": frame.site_name,
                        "body": frame.body_name,
                        "pos": frame.pos.tolist(),
                        "quat_wxyz": frame.quat.tolist(),
                    }
                    for frame in site_frames
                ),
                "output_frame": sensor.output_specs[0].frame,
                "actions": {
                    name: {
                        "norm_mm": float(np.linalg.norm(value)),
                        "mean_xyz_mm": value.mean(axis=(0, 1, 2, 3)).tolist(),
                    }
                    for name, value in outputs.items()
                },
                "slide_tangent_delta_norm_mm": slide_delta_norm,
                "slide_tangent_delta_mean_mm": slide_delta_mean.tolist(),
            }
        finally:
            if pose_plan is not None:
                pose_plan.close()
            if sensor is not None:
                sensor.close()
            backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "mjwarp"), default="mujoco")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.backend)
    rendered = json.dumps(result, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
