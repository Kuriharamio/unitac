# UniTac

UniTac is an independent tactile simulation plugin for [UniSim](https://github.com/unilabsim/unisim). It currently provides the HydroShear model. UniSim supplies detached collision geometry, sensor frames, lifecycle, and field metadata; UniTac owns the tactile model and does not retain a simulator backend.

## Supported

- HydroShear tactile displacement on CUDA, from caller-supplied closed gel geometry.
- GelSight Mini calibrated RGB, depth, and marker visualization.
- Verified letter press runs on UniSim MuJoCo and MJWarp backends.

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
```

HydroShear requires a working CUDA device and CUDA 12 compatible PyTorch and CuPy. In a checkout, install the runtime for each selected mode:

```bash
# MuJoCo physics and interactive/headless viewer
uv sync --extra hydroshear --extra mujoco --extra visualization

# MJWarp physics with the same viewer
uv sync --extra hydroshear --extra mujoco --extra mjwarp --extra visualization
```

`--no-render` does not need the `visualization` extra. The GelSight Mini calibration files and letter press geometry are included in the package.

## Run

Run one HydroShear sample from a static MuJoCo scene:

```bash
uv run --extra hydroshear --extra mujoco python examples/mujoco_backend.py
```

Run the same letter press sequence on either verified physics backend:

```bash
uv run python examples/letter_press/run.py --backend mujoco
uv run python examples/letter_press/run.py --backend mjwarp
```

The default opens an interactive viewer. Add `--headless` to record an MP4 and phase screenshots, or `--no-render` to run physics and HydroShear without loading the RGB calibration or creating a renderer. Both modes accept `--frames`, `--device`, and `--output-dir`; rendered modes also accept `--mode rgb|depth`. No-render logs a CUDA-reduced displacement summary; it omits depth and contact-pixel metrics that require visualization snapshots.

The selected UniSim backend advances physics and provides body poses, collision geometry, site frames, and playback snapshots. The viewer renders those snapshots with MuJoCo as a detached visualization model; it never advances MuJoCo physics for an MJWarp run. MuJoCo and MJWarp have completed full-cycle validation. The letter model's mesh collision is not currently compatible with Motrix's convex-hull collision projection; the remaining backends do not yet expose the required UniSim tactile geometry/site APIs. HydroShear does not feed contact forces back into the physics backend.

## Public API

`unisim.factory.create_tactile_sensor("unitac", model="hydroshear", ...)` returns a UniSim `TactileSensor`. Provide a closed gel triangle mesh in sensor-local metres, materialize it with a `TactileSceneDescription`, then call `update_pose`, `advance`, `read`, `reset`, and `close`. `read()` returns a view into a reusable CUDA Torch buffer; copy it before the next advance or reset if it must be retained. The minimal runnable integration is [examples/mujoco_backend.py](examples/mujoco_backend.py).

HydroShear geometry, SDF, and depth calculation require CUDA. The sensor has no CPU compute path. RGB and depth rendering are explicit CPU-side diagnostic work and are not performed by `read()`.

## Docs and license

- [Architecture](docs/en/architecture.md) · [架构说明](docs/zh/architecture.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [MIT License](LICENSE)
