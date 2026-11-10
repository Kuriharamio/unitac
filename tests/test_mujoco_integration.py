from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HydroShear requires CUDA")
@pytest.mark.parametrize("backend_type", ["mujoco", "mjwarp"])
def test_real_unisim_backend_drives_tactile_scene_and_pose_inputs(backend_type):
    pytest.importorskip("mujoco")
    pytest.importorskip("mjbatch")
    if backend_type == "mjwarp":
        pytest.importorskip("mujoco_warp")
    path = Path(__file__).parents[1] / "examples" / "mujoco_backend.py"
    spec = importlib.util.spec_from_file_location("unitac_mujoco_example", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load integration example at {path}")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    result = example.run(backend_type)

    assert result["backend"] == backend_type
    assert result["body_names"] == ("object", "left_pad", "right_pad", "static_fixture")
    assert result["pose_body_names"] == result["body_names"]
    assert "object_mesh_geom" in result["selected_object_geoms"]
    assert "excluded_aux_geom" in result["geom_names"]
    assert len(result["site_frames"]) == 2
    assert result["site_frames"][0]["site"] == "left_gel"
    assert result["site_frames"][0]["body"] == "left_pad"
    assert result["output_frame"] == "sensor_local"
    assert result["actions"]["no_contact"]["norm_model_units"] < 1e-5
    assert result["actions"]["press"]["norm_model_units"] > 1e-4
    assert result["slide_tangent_delta_norm_model_units"] > 1e-4
    assert result["slide_tangent_delta_mean_model_units"][0] > 0.0
    assert result["actions"]["retreat"]["norm_model_units"] < 1e-5
