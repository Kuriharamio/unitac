"""Small, lazy entry point for model-specific tactile visualizers."""

from __future__ import annotations

from typing import Any


def create_visualizer(model: str, **kwargs: Any) -> Any:
    """Create a visualizer owned by a UniTac tactile model.

    Optional GUI and image dependencies are imported only when a visualizer is
    explicitly requested; normal simulation and training imports stay light.
    """
    if model != "hydroshear":
        raise ValueError(f"unsupported UniTac visualization model: {model!r}")
    from .hydroshear.visualization import HydroShearVisualizer

    return HydroShearVisualizer(**kwargs)
