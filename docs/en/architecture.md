# UniTac architecture and validation

[English](architecture.md) | [中文](../zh/architecture.md)

## Boundaries

UniSim owns the model-neutral tactile contract. UniTac is an independent
UniSim plugin and owns HydroShear configuration, geometry preparation, CUDA
execution, history, output buffers, and resource cleanup. The HydroShear
repository is a numerical source only; it is not installed or imported at
runtime.

```text
unisim-core
└── tactile.py                 scene, capabilities, lifecycle, field metadata

unitac
├── __init__.py                light import and model dispatch
├── geometry.py                collision descriptor → body-local triangles
└── hydroshear/
    ├── config.py              HydroShear-only validated model settings
    ├── depth.py               camera rasterization and depth projection
    ├── sdf.py                 signed mesh distance
    ├── solver.py              HydroSoft history and marker displacement
    ├── fft_dilation.py        optional grid dilation path
    ├── fft_shear.py           optional dense shear path
    └── sensor.py              UniSim lifecycle and CUDA buffer ownership
```

## UniRay patterns

UniTac reuses UniRay's independent-plugin boundary and the same UniSim
principles: model packages are imported lazily; static topology is compiled
on a cold path; per-step state uses detached arrays or device tensors; output
buffer ownership and execution profiles are explicit; and the plugin never
holds a `SimBackend`. UniTac obtains geometry through UniSim's
`RayCollisionDescription` and mounts through `RaySiteFrameDescription`.

Tactile computation does not use a ray-caster interface. The contract permits
each model to choose its own field names, shape, units, coordinate frame, and
buffer lifetime. HydroShear declares a sensor-local displacement grid; a
different model may expose nodes, contact values, force, pressure, or images.

## HydroShear source migration

| Reference source at `26403177779c4f74689761c041fa9580b075dc08` | UniTac implementation | Retained behavior or change |
| --- | --- | --- |
| `src/hydroshear/depth.py` | `src/unitac/hydroshear/depth.py` | Camera projection, depth conventions, and triangle rasterization; persistent renderer workspaces replace per-call z/depth allocation. |
| `src/hydroshear/mesh_sdf.py` | `src/unitac/hydroshear/sdf.py` | Fused triangle-mesh signed distance and inside/outside classification; query/result storage is prepared for the fixed batch. |
| `src/hydroshear/sensors/hydrosoft.py` | `src/unitac/hydroshear/solver.py` | HydroSoft force history, friction response, dilation, shear, and reset semantics. History inputs are copied before reuse so persistent CuPy workspaces cannot mutate previous-frame state. |
| `src/hydroshear/sensors/fft_dilation.py` | `src/unitac/hydroshear/fft_dilation.py` | Optional FFT dilation implementation. |
| `src/hydroshear/sensors/fft_shear.py` | `src/unitac/hydroshear/fft_shear.py` | Optional dense shear implementation. |
| Old batch/configuration and viewer paths | `geometry.py`, `config.py`, `sensor.py` | Replaced by detached UniSim geometry, one validated HydroShear config, explicit lifecycle, preallocated buffers, and caller-stream Torch/CuPy interop. Old viewer and experiment setup are not runtime code. |

The geometry layer triangulates supported public mesh and primitive collision
descriptors, samples the object surface once with a fixed seed, and transforms
those samples into each pad frame on each step. HydroShear requires closed
object and gel surfaces, and production callers supply their licensed gel
mesh. The letter-press example bundles its collision mesh. GelSight Mini
calibration assets are loaded only when the model-specific visualizer is
explicitly created; the normal sensor training path does not initialize or
render them.

The optional FFT modules remain HydroShear implementation choices. The
default tested configuration uses the reference `use_fft_dilation=False` and
`use_matmul_shear=True` path. UniTac does not expose an FFT abstraction in the
UniSim contract.

## XENSIM architecture check

The read-only UniManip adapter and its vendored XENSIM source were inspected
at UniManip `main` and XENSIM snapshot
`006166e9dc28d74690c69f89c8d5920f449f6b3b`. The adapter constructs
`TactileFemBatch` from one body-local triangle mesh per object, environment
and pad counts, an optional FEM file, and solver options. It combines the
object and pad world poses per step, keeps results on CUDA, and maps solver
positions relative to FEM rest positions into a configured output field. The
underlying batch can return displacement, positions, force, and contact data;
the adapter currently publishes a configured `xensim` displacement field
with millimetre units. Reset is delegated to the FEM runtime and supports
selected environments.

| Shared by the UniTac contract | Model-specific in a future XENSIM plugin |
| --- | --- |
| Detached object collision geometry and body identity | FEM file, material/solver parameters, and FEM rest mesh |
| Sensor mount frames and per-environment world poses | FEM contact/depth mapping, matrix assembly, and iterative solves |
| Materialize, pose update, advance, read, reset, close | XENSIM channels, optional force/contact fields, and grid interpolation |
| Declared device profiles and each output's own metadata | XENSIM-specific cache policy and static FEM resources |

This audit supports reusing the scene, pose, lifecycle, capability, and field
metadata contracts. It does not justify adding FEM settings, fixed tactile
grids, or an XENSIM stub to UniSim or UniTac today. A future plugin would add
its FEM resources and solver under its own package and publish whatever field
shapes it actually produces.

## Numerical reference

`tests/fixtures/hydroshear_reference.npz` stores full marker arrays generated
with the source commit above on an RTX 4060 Laptop GPU (Torch 2.9.0+cu128,
CUDA 12.8). Both paths use synthetic closed box meshes, 2,048 fixed surface
samples, two pads, a 9×7 grid, float32 output in millimetres, and identical
solver settings. The fixtures contain no proprietary calibration assets.
`tests/test_hydroshear_reference.py` compares every field with
`rtol=2e-6, atol=2e-5`.

| Scenario | Reference maximum absolute displacement (mm) | Reference norm (mm) |
| --- | ---: | ---: |
| No contact | 0 | 0 |
| Press | 57.173409 | 427.700531 |
| Translation | 57.245087 | 451.158966 |
| Rotation | 57.159119 | 430.604095 |
| Sliding | 57.631008 | 466.962219 |
| Sustained contact | 57.173409 | 427.700531 |
| Two objects | 1.503018 | 12.737490 |
| Partial reset | 1.009494 | 9.361657 |

The single-environment reference field is `(2 pads, 9 rows, 7 columns,
3 components)`; UniTac adds the leading environment dimension to its runtime
field. The partial-reset fixture records the selected environment after reset.
All eight fixture comparisons and the pose-update/read-buffer lifecycle test
pass on the local CUDA device.
