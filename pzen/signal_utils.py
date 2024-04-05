import numpy as np


def normalize(array: np.ndarray) -> np.ndarray:
    """
    Normalize signal to stay within [-1, +1], but leave the zero point unmodified.
    """
    abs_max = np.abs(array).max()
    if abs_max != 0:
        return array / abs_max
    else:
        return array


def normalize_min_max(array: np.ndarray) -> np.ndarray:
    """
    This function normalizes the min/max to [-1, +1].

    Note that this can be a bit confusing, because it may look like the output has a constant
    DC-offset.
    """
    max = array.max()
    min = array.min()

    if max > min:
        return -1.0 + (array - min) / (max - min) * 2
    else:
        return array
