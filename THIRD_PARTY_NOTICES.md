HydroShear numerical components in `src/unitac/hydroshear/` are derived from
HydroShear, Copyright (c) 2026 MMintLab, under the MIT License. The source
reference is commit `26403177779c4f74689761c041fa9580b075dc08` on branch
`unimanip-api`. The license text is in `licenses/HydroShear-MIT.txt`.

The test and benchmark fixture
`tests/fixtures/gsmini_elastomer_collision_512.obj` is the original HydroShear
GelSight Mini collision mesh, Copyright (c) 2026 MMintLab, from the same MIT-
licensed source commit at
`src/hydroshear/assets/gsmini_elastomer_collision_512.obj`. Its SHA256 is
`93d97b732e1600ff4eff0e8bedf774a3f5630c06cd909871b06ab2ef649650db`. The raw
mesh is in the calibrated camera frame; benchmark code translates it by
`(+0.0005, +0.05975, +0.00725)` metres to the gel-local frame documented by the
source scene. No calibration arrays or images are redistributed. Production
callers may provide any closed gel triangle mesh they are authorized to use.
