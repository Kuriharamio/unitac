from __future__ import annotations

import numpy as np
import pytest

from unitac.hydroshear.config import HydroShearConfig, load_triangle_mesh

from .helpers import box_triangles, hydro_config


def test_hydroshear_config_validates_and_freezes_closed_gel_mesh():
    gel = box_triangles((0.010, 0.012625, 0.003))
    gel[:, :, 2] -= 0.003
    config = HydroShearConfig.from_mapping(hydro_config(gel))
    assert config.surface_points == 2048
    assert (config.rows, config.cols) == (9, 7)
    assert config.gel_triangles.shape == (12, 3, 3)
    assert not config.gel_triangles.flags.writeable


def test_gel_mesh_rejects_open_nonfinite_and_degenerate_triangles():
    closed_mesh = box_triangles((0.010, 0.012625, 0.003))
    with pytest.raises(ValueError, match="closed two-manifold"):
        load_triangle_mesh(closed_mesh[:-1])
    with pytest.raises(ValueError, match="finite"):
        load_triangle_mesh(np.full((12, 3, 3), np.nan, dtype=np.float32))
    degenerate = closed_mesh.copy()
    degenerate[0, 2] = degenerate[0, 1]
    with pytest.raises(ValueError, match="positive area"):
        load_triangle_mesh(degenerate)


def test_config_rejects_unknown_solver_fields_and_camera_out_of_bounds():
    gel = box_triangles((0.010, 0.012625, 0.003))
    with pytest.raises(ValueError, match="unknown HydroSoft solver parameters"):
        HydroShearConfig.from_mapping({"gel_mesh": gel, "solver": {"unknown": 1}})
    with pytest.raises(ValueError, match="outside the HydroShear camera view"):
        HydroShearConfig.from_mapping(
            {
                "gel_mesh": gel,
                "output_grid": {
                    "rows": 200,
                    "cols": 7,
                    "spacing_mm": [2.145, 2.153125],
                    "center_offset_mm": [-0.3575, -0.325],
                    "pixel_scale_mm": 0.065,
                },
            }
        )
