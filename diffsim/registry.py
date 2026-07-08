"""Typed registries with explicit duplicate and unknown-key failures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._builders: dict[str, Callable[..., T]] = {}

    def register(self, name: str, builder: Callable[..., T]) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError(f"{self.kind} name cannot be empty")
        if key in self._builders:
            raise ValueError(f"{self.kind} {key!r} is already registered")
        self._builders[key] = builder

    def create(self, name: str, *args, **kwargs) -> T:
        key = name.strip().lower()
        try:
            builder = self._builders[key]
        except KeyError as exc:
            choices = ", ".join(self.names()) or "<none>"
            raise ValueError(
                f"Unknown {self.kind} {name!r}. Available: {choices}"
            ) from exc
        return builder(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))
