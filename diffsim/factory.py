"""Environment construction boundary used by training entry points."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from .api import VectorEnv
from .registry import Registry


@dataclass(frozen=True)
class EnvBuildContext:
    args: Namespace
    device: torch.device
    control_dt: float
    focal_length: float
    scene_generator: Any = None
    extras: Mapping[str, Any] = field(default_factory=dict)


_ENVIRONMENTS: Registry[VectorEnv] = Registry("environment")
_BUILTINS_LOADED = False


def register_env(name: str, builder) -> None:
    _ENVIRONMENTS.register(name, builder)


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Import for registration only. Keeping this lazy avoids importing
    # PyTorch3D when callers use just the API/registry modules.
    from .envs import drone as _drone  # noqa: F401

    _BUILTINS_LOADED = True


def make_env(name: str, context: EnvBuildContext) -> VectorEnv:
    _load_builtins()
    return _ENVIRONMENTS.create(name, context)


def available_envs() -> tuple[str, ...]:
    _load_builtins()
    return _ENVIRONMENTS.names()
