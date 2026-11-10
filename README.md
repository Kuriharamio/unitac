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

HydroShear requires CUDA for signed-distance and depth computation. MuJoCo integration examples also need the `mujoco` extra. It accepts NumPy, CPU Torch, and CUDA Torch body poses, and returns a sensor-owned CUDA Torch displacement field. Gel triangles are supplied in the sensor frame and in metres. The letter press example bundles the MIT-licensed original 512-triangle collision mesh and the official Taxim calibration described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Letter press visualization

The HydroShear letter example uses the packaged `letter.STL` and GelSight Mini collision mesh in a real UniSim MuJoCo simulation. It runs a repeatable approach, press, hold, 30-degree rotation, lift, and return sequence. The default **Taxim-calibrated GelSight RGB** view uses Taxim's native 640×480, 0.0295 mm/pixel, 125-bin calibration; it is not genuine GelSight Mini-calibrated RGB. HydroShear depth and marker displacement are mapped in millimetres into the calibrated image plane without changing the solver or gel geometry. The default marker dots and arrows are a synthetic overlay driven by HydroShear's displacement output. `Space` pauses or resumes, `R` restarts, `N` or `.` takes one physics step, and `1`/`2`/`3` select RGB, marker displacement, or depth. `--headless` records an MP4 and writes the scene image, tactile image, and per-frame metrics.

Run with the bundled Taxim default:

```bash
uv run --extra hydroshear --extra mujoco python examples/letter_press/run.py
```

The interactive viewer opens at the overview angle with a movable free camera,
its left and right UI panels hidden, and the tactile image centered along the
right edge. The default arrow display gain is 10. Add `--headless` to record an
MP4 instead.

An optional external GelSight Mini profile is available for users who supply authorized TACSL files:

```bash
uv run --extra hydroshear --extra mujoco python examples/letter_press/run.py \
  --profile gelsight_mini --calibration-dir /path/to/authorized/gs_mini_data
```

That profile uses captured marker dots from its calibration imagery and adds synthetic arrows from HydroShear displacement. The display arrow gain changes only rendered vectors. The copied raw marker field and depth snapshot remain tied to the latest completed `advance()`; ordinary `read()` performs no visualization work.

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
