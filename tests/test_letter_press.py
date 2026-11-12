from __future__ import annotations

import json

import pytest
import torch
from examples.letter_press.runtime import run_no_render
from examples.letter_press.simulation import (
    CYCLE_FRAMES,
    PHASES,
    SIM_DT,
    LetterPressSimulation,
    phase_target,
    scene_files,
)
from unisim.backend.process_device import configure_backend_process_device
from unisim.factory import create_backend
from unisim.scene import SceneCfg


@pytest.mark.parametrize(
    ("frame", "expected_phase"),
    [(0, "Approach"), (20, "Press"), (60, "Hold"), (90, "Rotate"), (150, "Lift"), (190, "Return")],
)
def test_letter_press_phase_boundaries(frame: int, expected_phase: str) -> None:
    phase, target = phase_target(frame)

    assert phase == expected_phase
    assert target.shape == (2,)
    assert target.dtype.name == "float32"


def test_letter_press_cycle_has_one_reset_boundary() -> None:
    assert sum(duration for _, duration in PHASES) == CYCLE_FRAMES
    assert phase_target(CYCLE_FRAMES)[0] == "Approach"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HydroShear requires CUDA")
@pytest.mark.parametrize("backend_name", ("mujoco", "mjwarp", "motrix"))
def test_letter_press_no_render_smoke(backend_name: str, tmp_path, monkeypatch) -> None:
    pytest.importorskip("mujoco")
    if backend_name == "motrix":
        pytest.importorskip("motrixsim")
    else:
        pytest.importorskip("mjbatch")
    if backend_name == "mjwarp":
        pytest.importorskip("mujoco_warp")

    with scene_files() as xml_path:
        configure_backend_process_device(backend_name, "cuda:0")
        backend = create_backend(
            backend_name,
            SceneCfg(model_file=str(xml_path)),
            num_envs=1,
            sim_dt=SIM_DT,
            base_name="indenter",
            body_state_required=True,
        )
        with LetterPressSimulation(backend, xml_path, "cuda:0") as simulation:
            assert simulation.sensor is not None
            monkeypatch.setattr(
                simulation.sensor,
                "read_visualization_snapshot",
                lambda **_kwargs: pytest.fail("no-render must not read visualization snapshots"),
            )
            result = run_no_render(simulation, frames=2, output_dir=tmp_path)

    assert result["backend"] == backend_name
    assert result["frames"] == 2
    assert not list(tmp_path.glob("*.mp4"))
    assert not list(tmp_path.glob("*.png"))
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["phase"] for record in records] == ["Reset", "Approach", "Approach", "Reset"]
    assert all(record["finite"] for record in records)
    assert all(record["contact_pixels"] is None for record in records)
    assert all(record["peak_depth_mm"] is None for record in records)
    assert records[-1]["qpos"][-2:] == [0.0, 0.0]
