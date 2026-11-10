# UniTac

UniTac is an independent tactile-computation plugin for [UniSim](https://github.com/unilabsim/unisim). It currently implements HydroShear; the solver, mesh SDF, depth rasterizer, geometry processing, and temporal state are maintained here. The original HydroShear repository is not a runtime dependency.

The `unisim` package owns the model-neutral `TactileSensor` contract and lazy plugin factory. UniTac owns the HydroShear configuration and CUDA implementation. Models receive detached collision geometry and mounted site frames, then consume world body poses; they do not receive or retain a simulation backend.

## Install

The package is not published on PyPI yet. Install from `main`; the base import stays lazy, and the HydroShear extra adds Torch and CuPy for a CUDA 12 system:

```bash
uv add "unitac @ git+https://github.com/Kuriharamio/UniTac.git@main"
uv add "unitac[hydroshear] @ git+https://github.com/Kuriharamio/UniTac.git@main"
```

For repository development, clone with submodules and sync the locked environment:

```bash
git clone --recurse-submodules https://github.com/Kuriharamio/UniTac.git
uv sync --extra hydroshear --extra mujoco --extra viz
```

HydroShear requires CUDA for signed-distance and depth computation. MuJoCo integration examples also need the `mujoco` extra. It accepts NumPy, CPU Torch, and CUDA Torch body poses, and returns a sensor-owned CUDA Torch displacement field. Gel triangles are supplied in the sensor frame and in metres. The letter press example bundles the original 512-triangle collision mesh and the HydroShear upstream GelSight Mini calibration described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Letter press visualization

The HydroShear letter example uses the packaged `letter.STL` and GelSight Mini collision mesh in a real UniSim MuJoCo simulation. It runs a repeatable approach, press, hold, 30-degree rotation, lift, and return sequence. As in HydroShear, rigid MuJoCo collision on the gel is disabled; HydroSoft computes contact from the SDF at the commanded object pose. The marker grid uses the calibrated camera dot centers. The default **GelSight Mini-calibrated RGB** view follows HydroShear's GelSight Mini renderer: it center-crops and resizes depth to the calibrated active area, scales marker displacement by 0.05 display mm per HydroShear unit for dot motion, then overlays contact-only arrows at 4× display scale (minimum 1.5 pixels). The arrows use the sensor-local x/y field and are drawn at the displaced marker positions. With the Taxim profile, dots are synthetic overlays too. `Space` pauses or resumes, `R` restarts, `N` or `.` takes one physics step, `1` selects RGB with arrows, and `2` selects depth. `--headless` records an MP4 and writes the scene image, tactile image, and per-frame metrics.

Run with the bundled GelSight Mini calibration:

```bash
uv run --extra hydroshear --extra mujoco python examples/letter_press/run.py
```

The interactive viewer opens at the overview angle with a movable free camera,
its left and right UI panels hidden, and the tactile image centered along the
right edge. The default arrow display gain is 10. Add `--headless` to record an
MP4 instead. An authorized external GelSight Mini calibration can override the
bundled files with `--calibration-dir`. The explicit `--profile taxim` option
selects the separate Taxim calibration.

```bash
uv run --extra hydroshear --extra mujoco python examples/letter_press/run.py \
  --calibration-dir /path/to/authorized/gs_mini_data
```

The display arrow gain changes only rendered vectors. The copied raw marker field and depth snapshot remain tied to the latest completed `advance()`; ordinary `read()` performs no visualization work.

## Use

```python
from unisim.factory import create_tactile_sensor

sensor = create_tactile_sensor(
    "unitac",
    model="hydroshear",
    num_envs=64,
    config={
        "gel_mesh": licensed_closed_gel_triangles,  # (triangles, 3, 3), metres
        "surface_points": 2048,
        "output_grid": {
            "rows": 9,
            "cols": 7,
            "spacing_mm": [2.145, 2.153125],
            "pixel_scale_mm": 0.065,
        },
        "solver": {"use_matmul_shear": True},
        "device": "cuda:0",
    },
)
sensor.materialize(scene_description)
sensor.update_pose(body_pos, body_quat)  # world metres and unit wxyz
sensor.advance()  # one model step; history advances once
observation = sensor.read()["displacement"]
sensor.reset(env_ids=[3, 7])
sensor.close()
```

`read()` returns a view into a reusable buffer. Copy it before the next `advance`, `reset`, `rebuild`, or `close` if it must be retained. See [the architecture and evidence notes](docs/en/architecture.md).

Run the real MuJoCo backend integration sequence, or MJWarp when `mujoco_warp` is installed:

```bash
uv run --extra hydroshear --extra mujoco python examples/mujoco_backend.py
uv run --extra hydroshear --extra mujoco python examples/mujoco_backend.py --backend mjwarp
```

## Development

```bash
uv run ruff check .
uv run pytest -q
uv run python benchmarks/bench_gpu.py --envs 1 64 1024
```

HydroShear numerical source attribution and license are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/HydroShear-MIT.txt](licenses/HydroShear-MIT.txt).
