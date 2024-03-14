from pathlib import Path
from typing import Literal

import numpy as np
import soundfile
import torch

from .core_types import StrPath


def soundfile_write(
    path: StrPath,
    data: np.ndarray | torch.Tensor,
    sr: int,
    *,
    clip: Literal["normalize", "normalize_min_max", "ignore", "warn", "error"] = "warn",
) -> None:
    """
    Wrapper for soundfile.write with improvements:
    - allow normalization (apparently default behavior is to clip, see
      https://stackoverflow.com/q/69388531/1804173)
    - warn if clip
    - nan check
    - torch support
    - support pathlib and str
    - output path creation
    """

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    assert isinstance(data, np.ndarray)
    assert np.all(np.isfinite(data)), "Audio data contains NaNs or Infs."

    if clip == "normalize":
        data = normalize(data)

    elif clip == "normalize_min_max":
        data = normalize_min_max(data)

    else:
        num_clipped = int(np.sum(data > +1.0)) + int(np.sum(data < -1.0))
        if num_clipped > 0:
            msg = f"Audio data contains {num_clipped} clipped samples (out of {len(data)})"
            if clip == "error":
                raise ValueError(msg)
            elif clip == "warn":
                print(f"WARNING: {msg}")

    Path(path).parent.mkdir(exist_ok=True, parents=True)
    soundfile.write(str(path), data, sr)


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
