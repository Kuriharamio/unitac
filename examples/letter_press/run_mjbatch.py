"""Run the letter press example with UniSim's MJBatch-backed MuJoCo adapter."""

from __future__ import annotations

import json
from pathlib import Path

from unisim.backend.process_device import configure_backend_process_device
from unisim.factory import create_backend
from unisim.scene import SceneCfg

if __package__:
    from .runtime import (
        run_mujoco_playback_modes,
        standard_parser,
        validate_standard_args,
    )
    from .simulation import SIM_DT, LetterPressSimulation, scene_files
else:
    from runtime import (
        run_mujoco_playback_modes,
        standard_parser,
        validate_standard_args,
    )
    from simulation import SIM_DT, LetterPressSimulation, scene_files


def main() -> None:
    parser = standard_parser(__doc__, output_dir=Path("outputs/letter_press_mjbatch"))
    args = parser.parse_args()
    validate_standard_args(args)
    with scene_files() as xml_path:
        configure_backend_process_device("mujoco", args.device)
        backend = create_backend(
            "mujoco",
            SceneCfg(model_file=str(xml_path)),
            num_envs=1,
            sim_dt=SIM_DT,
            base_name="indenter",
            body_state_required=True,
        )
        with LetterPressSimulation(backend, xml_path, args.device) as simulation:
            result = run_mujoco_playback_modes(simulation, args)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
