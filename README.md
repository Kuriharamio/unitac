# UniTac

UniTac is an independent tactile simulation plugin for [UniSim](https://github.com/unilabsim/unisim). It currently provides the HydroShear model. UniSim supplies detached collision geometry, sensor frames, lifecycle, and field metadata; UniTac owns the tactile model and does not retain a simulator backend.

## Supported

- HydroShear tactile displacement on CUDA, from caller-supplied closed gel geometry.
- GelSight Mini calibrated RGB, depth, and marker visualization.
- UniSim MuJoCo integration; MJWarp is available with its optional extra.

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

HydroShear requires a working CUDA device and CUDA 12 compatible PyTorch and CuPy. The `visualization` extra installs OpenCV; `mujoco` installs MuJoCo integration dependencies; add `mjwarp` to use MJWarp. The GelSight Mini calibration files and letter press geometry are included in the package.

## Run

The complete public API integration example advances a MuJoCo scene and passes detached geometry and poses to HydroShear:

```bash
uv run --extra hydroshear --extra mujoco python examples/mujoco_backend.py
```

Run the letter press sequence with RGB, depth, and marker visualization:

```bash
uv run --extra hydroshear --extra visualization --extra mujoco \
  python examples/letter_press/run.py --headless --output-dir outputs/letter_press
```

The example advances MuJoCo poses through approach, press, hold, rotation, lift, and return. HydroShear computes contact from the object geometry and pose; it does not apply tactile contact forces back to MuJoCo. The example's arrow gain defaults to `40`. Use `--calibration-dir` to provide an external GelSight Mini calibration.

## Public API

`unisim.factory.create_tactile_sensor("unitac", model="hydroshear", ...)` returns a UniSim `TactileSensor`. Provide a closed gel triangle mesh in sensor-local metres, materialize it with a `TactileSceneDescription`, then call `update_pose`, `advance`, `read`, `reset`, and `close`. `read()` returns a view into a reusable CUDA Torch buffer; copy it before the next advance or reset if it must be retained. The full runnable integration is [examples/mujoco_backend.py](examples/mujoco_backend.py).

HydroShear geometry, SDF, and depth calculation require CUDA. The sensor has no CPU compute path. RGB and depth rendering are explicit CPU-side diagnostic work and are not performed by `read()`.

## Docs and license

- [Architecture](docs/en/architecture.md) · [架构说明](docs/zh/architecture.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [MIT License](LICENSE)
