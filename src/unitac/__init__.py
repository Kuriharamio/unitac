"""UniTac tactile simulation plugin for UniSim."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from unisim.optional import OptionalDependencyError

if TYPE_CHECKING:
    from unisim.tactile import TactileSensor

__version__ = "0.1.0"
__all__ = ["__version__", "create_tactile_sensor"]


def create_tactile_sensor(
    tactile_type: str,
    *,
    num_envs: int = 1,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> TactileSensor:
    """Create a selected UniTac tactile type; solver dependencies load on demand."""
    if tactile_type != "hydroshear":
        raise ValueError(f"unknown UniTac tactile type: {tactile_type!r}")
    if kwargs:
        raise TypeError(f"unexpected HydroShear factory arguments: {sorted(kwargs)}")
    if config is None:
        raise ValueError("HydroShear requires a tactile config with a closed gel_mesh")
    try:
        import cupy  # noqa: F401
        import torch  # noqa: F401
    except ImportError as error:
        raise OptionalDependencyError(
            "UniTac HydroShear requires its optional CUDA dependencies; install "
            "'unitac[hydroshear]' with a compatible PyTorch and CuPy CUDA runtime"
        ) from error
    from .hydroshear.sensor import HydroShearSensor

    return HydroShearSensor(num_envs=num_envs, config=config)
