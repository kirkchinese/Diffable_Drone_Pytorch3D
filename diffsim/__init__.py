"""Shared interfaces for differentiable robot simulation.

The package is introduced beside the legacy drone stack.  Existing drone
implementations remain the source of truth until their adapters have parity
tests; new robots should target these interfaces directly.
"""

from .api import (
    DifferentiableDynamics,
    EnvMetadata,
    FunctionalStep,
    StepOutput,
    VectorEnv,
)
from .factory import EnvBuildContext, available_envs, make_env

__all__ = [
    "DifferentiableDynamics",
    "EnvBuildContext",
    "EnvMetadata",
    "FunctionalStep",
    "StepOutput",
    "VectorEnv",
    "available_envs",
    "make_env",
]
