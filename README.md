# UniTac

UniTac is an independent tactile simulation plugin for [UniSim](https://github.com/unilabsim/unisim). It currently provides the HydroShear model. UniSim supplies detached collision geometry, sensor frames, lifecycle, and field metadata; UniTac owns the tactile model and does not retain a simulator backend.

## Supported

- HydroShear tactile displacement on CUDA, from caller-supplied closed gel geometry.
- GelSight Mini calibrated RGB, depth, and marker visualization.
- Verified letter press runs on UniSim MuJoCo, MJWarp, and Motrix backends.

HydroShear displacement is an empirical model output, not a physical length. Its field is shaped `(environment, sensor, row, column, xyz)` and reports `hydroshear_model_units`. Camera indentation depth is in millimetres. The visualizer maps model units to display motion with an empirical `0.05 mm/model unit` scale; arrow `display_gain` changes the arrows only.

## Install

UniTac is installed from Git:

```bash
uv add "unitac @ git+https://github.com/Kuriharamio/UniTac.git@main"
uv add "unitac[hydroshear] @ git+https://github.com/Kuriharamio/UniTac.git@main"
```

Install only the extras needed by the application:

```bash
uv add "unitac[hydroshear,visualization] @ git+https://github.com/Kuriharamio/UniTac.git@main"
uv add "unitac[hydroshear,mujoco] @ git+https://github.com/Kuriharamio/UniTac.git@main"
uv add "unitac[hydroshear,mujoco,mjwarp] @ git+https://github.com/Kuriharamio/UniTac.git@main"
uv add "unitac[hydroshear,motrix,visualization] @ git+https://github.com/Kuriharamio/UniTac.git@main"
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

`--no-render` does not need the `visualization` extra. Add it to run any tactile viewer. The GelSight Mini calibration files and letter press geometry are included in the package.

## Run

Run one HydroShear sample from a static MuJoCo scene:

```bash
uv run --extra hydroshear --extra mujoco python examples/mjbatch_backend.py
```

Each letter press entry point fixes its UniSim backend:

```bash
uv run python examples/letter_press/run_mjbatch.py
uv run python examples/letter_press/run_mjwarp.py
uv run python examples/letter_press/run_motrix.py
```

The MJBatch and MJWarp examples use MuJoCo playback for visualization. Their `--headless` option records an MP4 and phase screenshots; `--no-render` runs physics and HydroShear without loading RGB calibration data or creating a renderer. The Motrix example uses Motrix's native viewer and supports `--no-render`. All examples accept `--frames`, `--device`, and `--output-dir`; rendered modes also accept `--mode rgb|depth`. No-render logs a CUDA-reduced displacement summary; it omits depth and contact-pixel metrics that require visualization snapshots.

Each UniSim backend advances physics and provides body poses, collision geometry, and site frames. The MJBatch example uses UniSim's `mujoco` backend, which advances MuJoCo models through `mjbatch.Batch`; the MJWarp example renders public physics snapshots with MuJoCo but does not advance MuJoCo physics. The Motrix example renders through Motrix. All three backends have completed full-cycle no-render runs. Motrix supplies its compiled convex hull for the letter mesh, so HydroShear sees the same convex collision surface as Motrix. HydroShear does not feed contact forces back into the physics backend.

## Public API

`unisim.factory.create_tactile_sensor(tactile_type="hydroshear", ...)` returns a UniSim `TactileSensor`. Provide a closed gel triangle mesh in sensor-local metres, materialize it with a `TactileSceneDescription`, then call `update_pose`, `advance`, `read`, `reset`, and `close`. `read()` returns a view into a reusable CUDA Torch buffer; copy it before the next advance or reset if it must be retained. The minimal runnable integration is [examples/mjbatch_backend.py](examples/mjbatch_backend.py).

HydroShear geometry, SDF, and depth calculation require CUDA. The sensor has no CPU compute path. RGB and depth rendering are explicit CPU-side diagnostic work and are not performed by `read()`.

## Docs and license

- [Architecture](docs/en/architecture.md) · [架构说明](docs/zh/architecture.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [MIT License](LICENSE)
