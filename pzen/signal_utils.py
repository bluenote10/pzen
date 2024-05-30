from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import assert_never

# -----------------------------------------------------------------------------
# Low-level API
# -----------------------------------------------------------------------------


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize signal to stay within [-1, +1], but leave the zero point unmodified.
    """
    abs_max = np.abs(x).max()
    if abs_max != 0:
        return x / abs_max
    else:
        return x


def normalize_min_max(x: np.ndarray) -> np.ndarray:
    """
    This function normalizes the min/max to [-1, +1].

    Note that this can be a bit confusing, because it may look like the output has a constant
    DC-offset.
    """
    max = x.max()
    min = x.min()

    if max > min:
        return -1.0 + (x - min) / (max - min) * 2
    else:
        return x


def pad_to_multiple_of(x: np.ndarray, block_size: int) -> np.ndarray:
    """
    Right-pad signal so that its length is a multiple of `block_size`.
    """
    assert block_size > 0
    n_pad = (block_size - len(x) % block_size) % block_size
    x = np.pad(x, (0, n_pad))
    return x


# -----------------------------------------------------------------------------
# High-level API
# -----------------------------------------------------------------------------


@dataclass
class Samples:
    value: int


@dataclass
class Seconds:
    value: float


"""
Note on naming: Alternative names considered 'Duration' or 'Length'. While most usages
actually are semantically more a "duration" or "length" there can also be usages that
are semantically an "offset", which would make duration/length awkward. `t: Time` is
short and works reasonable well for both.
"""
Time = Samples | Seconds


def in_samples(t: Time | None, sr: int) -> int:
    if t is None:
        return 0
    elif isinstance(t, Samples):
        return t.value
    elif isinstance(t, Seconds):
        return round(t.value * sr)
    else:
        assert_never(t)


@dataclass
class Signal:
    """
    High-level wrapper of signals.

    Note on naming: Alternative names considered 'Audio' or 'Wave'. However, this class
    is more general than audio/wave signals, because it could also be used for other
    kind of signals like amplitude envelops, frequency sequences, phases, etc.
    """

    x: np.ndarray
    sr: int

    def __post_init__(self) -> None:
        # Invariants
        assert self.x.ndim == 1, f"Signals must have a dimension of 1, but is {self.x.ndim}"
        assert self.sr > 0

    # Construction helpers

    @staticmethod
    def empty(sr: int) -> Signal:
        return Signal(x=np.array([]), sr=sr)

    @staticmethod
    def zeros(t: Time, sr: int) -> Signal:
        n = in_samples(t, sr)
        return Signal(x=np.zeros(n), sr=sr)

    # Basic operators

    def __len__(self) -> int:
        return len(self.x)

    def __add__(self, other: Signal) -> Signal:
        assert (
            self.sr == other.sr
        ), f"Can only add signals of same sample rate, but {self.sr} != {other.sr}"
        assert len(self) == len(other), (
            f"Trying to add signals with different lengths ({len(self)} != {len(other)}). "
            "Use `mix_at` if implicit length extension is desired."
        )
        return Signal(x=self.x + other.x, sr=self.sr)

    def __mul__(self, other: Signal) -> Signal:
        assert (
            self.sr == other.sr
        ), f"Can only multiply signals of same sample rate, but {self.sr} != {other.sr}"
        assert len(self) == len(
            other
        ), f"Trying to multiply signals with different lengths ({len(self)} != {len(other)})."
        return Signal(x=self.x * other.x, sr=self.sr)

    # Misc

    @property
    def unpack(self) -> tuple[np.ndarray, int]:
        return self.x, self.sr

    def in_samples(self, t: Time | None) -> int:
        return in_samples(t, self.sr)

    def len(self) -> int:
        """Convenience alternative to __len__ for more fluent API."""
        return len(self)

    # Unary operations

    def scale(self, factor: float) -> Signal:
        return Signal(factor * self.x, self.sr)

    def abs(self) -> Signal:
        return Signal(np.abs(self.x), self.sr)

    def square(self) -> Signal:
        return Signal(self.x * self.x, self.sr)

    def normalize(self) -> Signal:
        return Signal(normalize(self.x), self.sr)

    def normalize_min_max(self) -> Signal:
        return Signal(normalize_min_max(self.x), self.sr)

    # Length affecting operations

    def pad(self, *, pad_l: Time | None = None, pad_r: Time | None = None) -> Signal:
        n_pad_l = self.in_samples(pad_l)
        n_pad_r = self.in_samples(pad_r)
        x = np.pad(self.x, (n_pad_l, n_pad_r))
        return Signal(x, self.sr)

    def pad_to_multiple_of(self, block_size: Time) -> Signal:
        x = pad_to_multiple_of(self.x, block_size=self.in_samples(block_size))
        return Signal(x, self.sr)

    # Advanced operations

    def mix_at(self, offset: Time, other: Signal, allow_extend: bool = True) -> Signal:
        assert (
            self.sr == other.sr
        ), f"Can only mix signals of same sample rate, but {self.sr} != {other.sr}"

        offset_index = self.in_samples(offset)
        if offset_index < 0:
            raise ValueError(f"Negative offset indices ({offset_index}) are not supported")
        required_len = offset_index + other.len()

        if required_len > self.len():
            if not allow_extend:
                raise ValueError(
                    f"Mixing signal would require a length of {required_len}, "
                    f"but signal only has length {self.len()}"
                )
            x = np.pad(self.x, (0, required_len - self.len()))
        else:
            x = self.x.copy()

        x[offset_index : offset_index + other.len()] += other.x
        return Signal(x=x, sr=self.sr)
