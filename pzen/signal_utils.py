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
class Audio:
    x: np.ndarray
    sr: int

    @staticmethod
    def empty(sr: int) -> Audio:
        return Audio(x=np.array([]), sr=sr)

    @staticmethod
    def silence(t: Time, sr: int) -> Audio:
        n = in_samples(t, sr)
        return Audio(x=np.zeros(n), sr=sr)

    def __len__(self) -> int:
        return len(self.x)

    def __post_init__(self) -> None:
        assert self.x.ndim == 1, f"Audio signals must have a dimension of 1, but is {self.x.ndim}"

    def __add__(self, other: Audio) -> Audio:
        assert (
            self.sr == other.sr
        ), f"Can only add audio signals of same sample rate, but {self.sr} != {other.sr}"
        assert len(self) == len(other), (
            f"Trying to add audio signals with different lengths ({len(self)} != {len(other)}). "
            "Use `mix_at` if implicit length extension is desired."
        )
        return Audio(x=self.x + other.x, sr=self.sr)

    def in_samples(self, t: Time | None) -> int:
        return in_samples(t, self.sr)

    def len(self) -> int:
        return len(self)

    def scale(self, factor: float) -> Audio:
        return Audio(factor * self.x, self.sr)

    def normalize(self) -> Audio:
        return Audio(normalize(self.x), self.sr)

    def normalize_min_max(self) -> Audio:
        return Audio(normalize_min_max(self.x), self.sr)

    def pad(self, *, pad_l: Time | None = None, pad_r: Time | None = None) -> Audio:
        n_pad_l = self.in_samples(pad_l)
        n_pad_r = self.in_samples(pad_r)
        x = np.pad(self.x, (n_pad_l, n_pad_r))
        return Audio(x, self.sr)

    def pad_to_multiple_of(self, block_size: Time) -> Audio:
        x = pad_to_multiple_of(self.x, block_size=self.in_samples(block_size))
        return Audio(x, self.sr)

    def mix_at(self, offset: Time, other: Audio, allow_extend: bool = True) -> Audio:
        assert (
            self.sr == other.sr
        ), f"Can only mix audio signals of same sample rate, but {self.sr} != {other.sr}"

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
        return Audio(x=x, sr=self.sr)
