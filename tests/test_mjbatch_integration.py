from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HydroShear requires CUDA")
def test_static_mjbatch_example():
    pytest.importorskip("mujoco")
    pytest.importorskip("mjbatch")
    path = Path(__file__).parents[1] / "examples" / "mjbatch_backend.py"
    spec = importlib.util.spec_from_file_location("unitac_mujoco_example", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load integration example at {path}")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    displacement = example.run()

    assert displacement.shape == (1, 1, 9, 7, 3)
    assert torch.isfinite(torch.from_numpy(displacement)).all()
