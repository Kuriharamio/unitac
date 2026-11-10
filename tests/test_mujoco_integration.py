from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HydroShear requires CUDA")
@pytest.mark.parametrize("backend_type", ["mujoco", "mjwarp"])
def test_static_unisim_backend_example(backend_type):
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

    displacement = example.run(backend_type)

    assert displacement.shape == (1, 1, 9, 7, 3)
    assert torch.isfinite(torch.from_numpy(displacement)).all()
