import numpy as np
from scipy.signal import find_peaks as find_peaks_orig


def assert_same_shape(*arrays: np.ndarray) -> None:
    shapes = [x.shape for x in arrays]
    shape_set = set(shapes)
    assert (
        len(shape_set) == 1
    ), f"Expected all tensors to have the same shape, but shapes are: {shapes}"


def find_peaks(x: np.ndarray, distance: int, ignore_peaks_at_boundary: bool = False) -> np.ndarray:
    """
    Improved `find_peaks` including a work-around for the strange handling of neighborhoods:
    https://github.com/scipy/scipy/issues/18495

    https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
    """

    peak_indices, _properties = find_peaks_orig(x, distance=distance)

    selected_peak_indices = []

    # To fix the semantics of `find_peaks` we manually check the dominance over the local
    # neighborhood.
    #
    # Note: The reason not to use `argrelmax` was that it does not detect peaks if a peak repeats
    # the same value multiple times. Apparently such peaks are not at all considered peaks by
    # `argrelmax`. In my use cases that would be fatal, because there could be strong "peaks",
    # which happen to repeat the same value just due to noise. They definitely should be detected
    # anyway. See reference:
    # https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.signal.argrelmax.html

    for peak_idx in peak_indices:
        if ignore_peaks_at_boundary:
            # Ignore peaks if they are exactly at the boundary
            if peak_idx == 0 or peak_idx == len(x) - 1:
                continue

        neighborhood_left = np.s_[max(peak_idx - distance, 0) : peak_idx]
        neighborhood_right = np.s_[peak_idx + 1 : peak_idx + distance + 1]
        neighborhood_max_left = np.max(x[neighborhood_left])
        neighborhood_max_right = np.max(x[neighborhood_right])

        # If a peak does not satisfy the local dominance criterion we don't even consider
        # it a peak.
        if x[peak_idx] < neighborhood_max_left or x[peak_idx] < neighborhood_max_right:
            continue

        selected_peak_indices.append(peak_idx)

    return np.array(selected_peak_indices)


def expspace(value_from: float, value_upto: float, n: int, grow_factor: float) -> np.ndarray:
    """
    Function similar to np.linspace / np.logspace, but with a slightly more convenient interface
    than logspace: It allows to specify a `grow_factor` which determines how much larger/smaller
    each interval is to the next one.

    For instance, if the `grow_factor` is 0.5, the second interval is half the size of the first,
    the third half the size of the second, etc.

    For lack of a better word, called 'expspace'...
    """
    if n <= 1:
        raise ValueError(f"Number of points must be >= 2, got: {n}")

    scaled = np.cumsum(np.cumprod(np.full(n, grow_factor)))

    scaled_min = np.min(scaled)
    scaled_max = np.max(scaled)
    delta_scaled = scaled_max - scaled_min
    delta_target = value_upto - value_from
    return (scaled - scaled_min) / delta_scaled * delta_target + value_from
