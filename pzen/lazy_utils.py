from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class Lazy(Generic[T]):

    def __init__(self, factory: Callable[[], T]):
        self._factory: Callable[[], T] = factory
        self._value: T | None = None

    def __call__(self) -> T:
        if self._value is None:
            self._value = self._factory()
        return self._value

    @staticmethod
    def get(x: T | Lazy[T]) -> T:
        if isinstance(x, Lazy):
            return x()
        else:
            return x
