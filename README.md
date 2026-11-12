# UniTac

UniTac is an independent tactile simulation plugin for [UniSim](https://github.com/unilabsim/unisim). It currently provides the HydroShear model. UniSim supplies detached collision geometry, sensor frames, lifecycle, and field metadata; UniTac owns the tactile model and does not retain a simulator backend.

## Supported

- HydroShear tactile displacement on CUDA, from caller-supplied closed gel geometry.
- GelSight Mini calibrated RGB, depth, and marker visualization.
- Verified complete example workflows on UniSim MuJoCo, MJWarp, and Motrix backends.

HydroShear displacement is an empirical model output, not a physical length. Its field is shaped `(environment, sensor, row, column, xyz)` and reports `hydroshear_model_units`. Camera indentation depth is in millimetres. The visualizer maps model units to display motion with an empirical `0.05 mm/model unit` scale.

## Install

UniTac is installed from Git:

```bash
uv add "unitac @ git+https://github.com/Kuriharamio/UniTac.git@main"
uv add "unitac[hydroshear] @ git+https://github.com/Kuriharamio/UniTac.git@main"
```

HydroShear requires a working CUDA device and CUDA 12 compatible PyTorch and CuPy. In a checkout, install the runtime for each selected mode:

```bash
# MJBatch-backed MuJoCo physics and interactive/headless viewer
uv sync --extra hydroshear --extra mujoco --extra visualization

# MJWarp physics with the same viewer
uv sync --extra hydroshear --extra mujoco --extra mjwarp --extra visualization

# Motrix physics with its native viewer
uv sync --extra hydroshear --extra motrix --extra visualization
```

## Run

A minimal public API example reads the initial poses provided by UniSim and computes one HydroShear sample:

```bash
uv run --extra hydroshear --extra mujoco python examples/mjbatch_backend.py
```

The letter press entry points run the complete physics simulation and tactile update workflow:

```bash
uv run python examples/letter_press/run_mjbatch.py
uv run python examples/letter_press/run_mjwarp.py
uv run python examples/letter_press/run_motrix.py
```

The `--headless` option records an MP4 and phase screenshots. The `--no-render` option runs physics and HydroShear without loading RGB calibration data or creating a renderer. The Motrix example uses Motrix's native viewer and supports `--no-render`.

## Public API

`unisim.factory.create_tactile_sensor(tactile_type="hydroshear", ...)` returns a UniSim `TactileSensor`. Provide a closed gel triangle mesh in sensor-local metres, materialize it with a `TactileSceneDescription`, then call `update_pose`, `advance`, `read`, `reset`, and `close`. `read()` returns a view into a reusable CUDA Torch buffer. The minimal public API example is [examples/mjbatch_backend.py](examples/mjbatch_backend.py).

## Docs and license

- [Architecture](docs/en/architecture.md) · [架构说明](docs/zh/architecture.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [MIT License](LICENSE)
