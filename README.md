# UniTac

UniTac is an independent tactile-computation plugin for [UniSim](https://github.com/unilabsim/unisim). It currently implements HydroShear; the solver, mesh SDF, depth rasterizer, geometry processing, and temporal state are maintained here. The original HydroShear repository is not a runtime dependency.

The `unisim` package owns the model-neutral `TactileSensor` contract and lazy plugin factory. UniTac owns the HydroShear configuration and CUDA implementation. Models receive detached collision geometry and mounted site frames, then consume world body poses; they do not receive or retain a simulation backend.

## Install

The base package imports without Torch or CUDA libraries. Install the solver extra on a CUDA 12 system:

```bash
uv add unitac
uv add unitac --extra hydroshear
```

For repository development, clone with submodules and sync the locked environment:

```bash
git clone --recurse-submodules https://github.com/Kuriharamio/UniTac.git
uv sync --extra hydroshear --extra viz
```

HydroShear requires CUDA for signed-distance and depth computation. It accepts NumPy, CPU Torch, and CUDA Torch body poses, and returns a sensor-owned CUDA Torch displacement field. Gel triangles are supplied by the caller in the sensor frame and in metres. No HydroShear calibration files or gel mesh are bundled.

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
sensor.advance()                         # one model step; history advances once
observation = sensor.read()["displacement"]
sensor.reset(env_ids=[3, 7])
sensor.close()
```

`read()` returns a view into a reusable buffer. Copy it before the next `advance`, `reset`, `rebuild`, or `close` if it must be retained. See [the architecture and evidence notes](docs/en/architecture.md) and the [press-and-slide example](examples/press_slide.py).

Run the example and save its marker-field figure:

```bash
uv run --extra hydroshear --extra viz python examples/press_slide.py
```

## Development

```bash
uv run ruff check .
uv run pytest -q
uv run python benchmarks/bench_gpu.py --envs 1 64 1024
```

HydroShear numerical source attribution and license are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/HydroShear-MIT.txt](licenses/HydroShear-MIT.txt).
