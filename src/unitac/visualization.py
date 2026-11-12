"""Small, lazy entry point for tactile-type-specific visualizers."""

from __future__ import annotations

from typing import Any


def create_visualizer(tactile_type: str, **kwargs: Any) -> Any:
    """Create a visualizer for a UniTac tactile type.

    Optional GUI and image dependencies are imported only when a visualizer is
    explicitly requested; normal simulation and training imports stay light.
    """
    if tactile_type != "hydroshear":
        raise ValueError(f"unsupported UniTac visualization tactile type: {tactile_type!r}")
    from .hydroshear.visualization import HydroShearVisualizer

    return HydroShearVisualizer(**kwargs)
