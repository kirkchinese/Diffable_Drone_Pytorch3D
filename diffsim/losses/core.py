"""Reusable differentiable loss composition without host synchronization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import torch
from torch import nn


@dataclass
class LossOutput:
    loss: torch.Tensor
    terms: Mapping[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: Mapping[str, torch.Tensor] = field(default_factory=dict)


class LossTerm(Protocol):
    def __call__(self, context: Any) -> torch.Tensor: ...


class WeightedLossComposer(nn.Module):
    """Combine named scalar tensor losses while retaining every raw term.

    Terms must return tensors. Conversion to Python numbers belongs in logging,
    never in the differentiable hot path.
    """

    def __init__(self, terms: Mapping[str, LossTerm], weights: Mapping[str, float]):
        super().__init__()
        unknown = set(weights) - set(terms)
        if unknown:
            raise ValueError(f"Weights supplied for unknown loss terms: {sorted(unknown)}")
        self.terms = dict(terms)
        self.weights = {name: float(weights.get(name, 1.0)) for name in terms}

    def forward(self, context: Any) -> LossOutput:
        raw = {name: term(context) for name, term in self.terms.items()}
        if not raw:
            raise ValueError("At least one loss term is required")
        for name, value in raw.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise TypeError(f"Loss term {name!r} must return a scalar tensor")
        total = sum(value * self.weights[name] for name, value in raw.items())
        return LossOutput(loss=total, terms=raw)
