"""Small, tensor-first contracts shared by drone and quadruped simulators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Protocol, TypeVar, runtime_checkable

import torch


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class EnvMetadata:
    """Static capabilities; never place per-step tensors in this object."""

    name: str
    batch_size: int
    device: torch.device
    physics_dt: float
    control_dt: float
    differentiable_dynamics: bool = True
    differentiable_observation: bool = False


@dataclass
class StepOutput(Generic[StateT, ObservationT]):
    """Result of a vectorized environment transition."""

    state: StateT
    observation: ObservationT | None = None
    terminated: torch.Tensor | None = None
    diagnostics: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class FunctionalStep(Generic[StateT]):
    """Pure dynamics result used by new differentiable physics backends."""

    state: StateT
    diagnostics: Mapping[str, torch.Tensor] = field(default_factory=dict)


@runtime_checkable
class DifferentiableDynamics(Protocol[StateT, ActionT]):
    """Functional hot-loop physics: no hidden per-environment mutation."""

    def step(self, state: StateT, action: ActionT, dt: float) -> FunctionalStep[StateT]: ...


@runtime_checkable
class VectorEnv(Protocol[ActionT]):
    """Minimal compatibility contract for batched training environments.

    The legacy drone environment already satisfies the reset/step portion.  New
    environments additionally expose ``metadata`` and should keep all hot-path
    tensors on ``metadata.device``.
    """

    B: int
    device: torch.device

    def reset(self) -> Any: ...

    def step(self, action: ActionT, **kwargs: Any) -> Any: ...
