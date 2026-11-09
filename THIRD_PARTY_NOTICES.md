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
source scene. The same file is used by the letter press example.

`src/unitac/hydroshear/assets/letter_press/letter.STL` is the HydroShear
letter mesh from `Kuriharamio/hydroshear` commit
`0eff954e2ac538e55abd58e278c2574311bc8687` (`assets/mesh/letter.STL`),
SHA256 `562292fa4e38d310ecd07505ab1fe08424d62e029d4e7c625dc6a6980b070e7a`.
The source repository's MIT license applies; the license text is included in
`licenses/HydroShear-MIT.txt`.

The GelSight Mini calibration archive and background images are not bundled:
the HydroShear source repository directs users to obtain these TACSL assets
separately, and no redistribution license for the three files has been
verified. The files were introduced at HydroShear commit
`460c94aea77a5b5e862962f7e228e025e5ac71a6` on `perf/sdf`:

- `polycalib.npz`, SHA256
  `34937ac09579b259fd344f99fbc71ebb41a6ad8bbe5f75118d016ada8fbb044a`.
- `mmint_gs_mini_2D9C-N227_07072025_bg_R_no_dots.png`, SHA256
  `aaa0fdbfb7bb35a1223a4d1120b20b2a2bcdacb4030a74d22756b6bc46264cad`.
- `mmint_bg.jpg`, SHA256
  `1da75bd77c00a41b18e6de974bbcd4f6df532c07d6a9e0e618597467687bdd55`.

The [TACSL setup instructions](https://github.com/isaac-sim/IsaacGymEnvs/blob/tacsl/isaacgymenvs/tacsl_sensors/install/tacsl_setup.md)
instruct users to download GelSight assets separately. Supply an authorized
directory containing the three files above at runtime. Production callers
may provide any closed gel triangle mesh they are authorized to use.
