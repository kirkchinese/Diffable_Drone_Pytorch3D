"""Composable loss contracts and legacy adapters."""

from .core import LossOutput, LossTerm, WeightedLossComposer
from .factory import LossBuildContext, available_losses, make_loss

__all__ = [
    "LossBuildContext",
    "LossOutput",
    "LossTerm",
    "WeightedLossComposer",
    "available_losses",
    "make_loss",
]
