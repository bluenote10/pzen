import numpy as np


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


def pad_to_blocksize(x: np.ndarray, block_size: int) -> tuple[np.ndarray, int]:
    """
    Pad signal so that its length is a multiple of `block_size`.
    """
    assert block_size > 0
    n_pad = (block_size - len(x) % block_size) % block_size
    x = np.pad(x, (0, n_pad))
    return x, n_pad
