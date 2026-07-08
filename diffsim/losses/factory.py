"""Loss construction boundary; legacy losses stay numerically untouched."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

from ..registry import Registry


@dataclass(frozen=True)
class LossBuildContext:
    args: Namespace
    control_dt: float


_LOSSES = Registry("loss")
_BUILTINS_LOADED = False


def register_loss(name: str, builder) -> None:
    _LOSSES.register(name, builder)


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from . import drone as _drone  # noqa: F401
    from . import go2 as _go2  # noqa: F401

    _BUILTINS_LOADED = True


def make_loss(name: str, context: LossBuildContext):
    _load_builtins()
    return _LOSSES.create(name, context)


def available_losses() -> tuple[str, ...]:
    _load_builtins()
    return _LOSSES.names()
