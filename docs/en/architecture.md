# Architecture

[English](architecture.md) | [中文](../zh/architecture.md)

## Boundaries

```text
UniSim
  public tactile scene, pose, lifecycle, and field contracts
    ↓
UniTac
  model selection and HydroShear implementation
    ↓
HydroShear
  geometry sampling, depth, SDF, temporal state, and displacement
```

UniSim owns the backend-neutral `TactileSensor`, `TactileSceneDescription`, and `TactileFieldSpec` contracts. UniTac receives detached collision geometry, mounted sensor frames, and world body poses. It does not retain a simulator backend. HydroShear owns its configuration, CUDA computation, buffers, reset, rebuild, and cleanup. `src/unitac/__init__.py` selects the model with a small factory branch.

## Data flow

1. `materialize(scene)` validates object and sensor identities, converts supported collision descriptors to body-local triangles, samples the configured object surfaces, and allocates device workspaces.
2. `update_pose(body_pos, body_quat)` replaces cached world poses. Positions are in metres and quaternions use unit `wxyz` order.
3. `advance()` transforms object samples into each sensor frame, rasterizes indentation depth in millimetres, converts marker heights to metres for HydroSoft, advances its temporal state once, and writes the marker field.
4. `read()` returns the latest marker field without advancing state. `read_visualization_snapshot()` explicitly copies diagnostic arrays to CPU for the visualizer.

## Lifecycle and outputs

The caller owns physics stepping and calls `materialize`, `update_pose`, `advance`, `read`, `reset`, and `close`. Geometry changes use `rebuild(scene)`. The model receives no backend object and does not apply contact forces back to the simulator.

The `displacement` field is float32 CUDA Torch data shaped `(environment, sensor, row, column, xyz)`. Its `TactileFieldSpec.units` is `hydroshear_model_units`: HydroSoft's empirical coefficients do not define a physical displacement length. `read()` returns a view into a sensor-owned buffer that can be reused at the next advance, reset, rebuild, or close.

Visualization snapshots contain copied sensor-local marker positions in millimetres, camera indentation depth in millimetres with shape `(sensor, 175, 100)`, and the unscaled `displacement` model field. GelSight Mini rendering converts depth millimetres to calibrated height pixels by dividing by `0.065 mm/pixel`. It maps HydroShear model units to display motion with the empirical `0.05 mm/model-unit` coefficient. These display values are not physical displacement measurements. RGB calibration and OpenCV are loaded only when the visualizer is created.

## Configuration and devices

The HydroShear config requires `gel_mesh`, a closed `(triangles, 3, 3)` mesh in sensor-local metres. `surface_points` defaults to 2,048. `output_grid` defines marker rows, columns, spacing, camera centers, and pixel scale. `solver` contains HydroSoft parameters and optional FFT or matrix multiplication paths. `device` selects a CUDA device.

CUDA is required for geometry distance and depth computation. NumPy and CPU Torch poses are copied to the configured device. CUDA Torch poses use the caller's current stream and DLPack interoperability. Normal `read()` stays on CUDA; diagnostic snapshots and image rendering perform explicit device-to-host copies.

## Adding a model

Add a model-specific implementation and one selection branch in the UniTac factory. Implement UniSim's `TactileSensor` contract and declare each output's shape, units, frame, device profile, and buffer lifetime. A model can use its own geometry requirements and output fields; UniSim does not prescribe HydroShear's grid or displacement format.
