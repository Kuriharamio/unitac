from __future__ import annotations

import json

import pytest
import torch
from examples.letter_press.run import run_no_render
from examples.letter_press.simulation import (
    CYCLE_FRAMES,
    PHASES,
    LetterPressSimulation,
    phase_target,
    scene_files,
)


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
@pytest.mark.parametrize("backend_name", ("mujoco", "mjwarp"))
def test_letter_press_no_render_smoke(backend_name: str, tmp_path, monkeypatch) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("mjbatch")
    if backend_name == "mjwarp":
        pytest.importorskip("mujoco_warp")

    with scene_files() as xml_path:
        with LetterPressSimulation(backend_name, xml_path, "cuda:0") as simulation:
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
    assert records[-1]["qpos"] == [0.0, 0.0]
