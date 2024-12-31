from typing import Callable

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


def expspace_delta(value_from: float, value_upto: float, n: int, delta: float) -> np.ndarray:
    """
    Similar to the function above, but allows to specify the size of the first interval (`delta`)
    instead of the growth factor.

    The idea here is to solve this equation for the growth factor `f` given a delta `d`, and a
    range `r`:

        n - 1
         ___
         ╲         i
         ╱    d ⋅ f  = r
         ‾‾‾
        i = 0

    We can pull out the `d`

            n - 1
             ___
             ╲     i
        d ⋅  ╱    f  = r
             ‾‾‾
            i = 0

    bring it to the other side, and write it using `c`.

            r
        c = ─
            d

    The sum then can be rewritten using the formula for the geometric series.

             N
        1 - f
        ────── = c
         1 - f

    We now have to solve for `f`, which unfortunately involves a higher-order polynomial.
    According to the following this may be solvable analytically, but a numeric solution
    should be good enough for now:

    https://math.stackexchange.com/questions/1437161/how-to-solve-nth-degree-polynomial-equation-with-terms-only-in-n-1-and-0

    (Math written with: https://arthursonzogni.com/Diagon/#Math)
    """

    value_range = value_upto - value_from

    if n <= 1:
        raise ValueError(f"Number of points must be >= 2, got: {n}")
    elif n == 2:
        if delta == value_range:
            return np.array([value_from, value_upto])
        else:
            raise ValueError("For n = 1, the delta must match the value range")

    if delta == 0.0:
        raise ValueError("Delta must not be zero.")

    if value_range > 0.0:
        if delta < 0.0:
            raise ValueError("Delta must not be negative, if range is positive.")
    elif value_range < 0.0:
        if delta > 0.0:
            raise ValueError("Delta must not be positive, if range is negative.")
    else:
        raise ValueError("Value range must not be zero.")

    if abs(delta) >= abs(value_range):
        raise ValueError("Delta must not be larger than value range.")

    c = value_range / delta
    assert c > 1.0

    # In terms of the geometric series we care about the number of terms/interval,
    # not the number of points to generate.
    N = n - 1

    def f(x: float) -> float | None:
        # Return the value of geometric series up to term N.
        if x == 1.0:
            return N
        else:
            try:
                return (1 - x**N) / (1 - x)
            except OverflowError:
                return None

    ratio = delta / (value_range / N)
    if ratio < 1.0:
        # Growing case
        x_l = 1.0
        x_r = c
        # Evaluating the geometric series for large factors can overflow in the `x**N`
        # term. We bring the factor down closer to 1.0 via square-rooting until we have
        # a valid value.
        while f(x_r) is None:
            x_r = np.sqrt(x_r)
    elif ratio > 1.0:
        # Shrinking case
        x_l = 0.0
        x_r = 1.0
    else:
        return expspace(value_from, value_upto, n, grow_factor=1.0)

    f_l = f(x_l)
    f_r = f(x_r)
    assert f_l is not None
    assert f_r is not None
    assert f_l - c < 0.0, f"Expected `f(x_l) - c` to be negative, but is {f_l - c} ({x_l=}, {c=})"
    assert f_r - c > 0.0, f"Expected `f(x_r) - c` to be positive, but is {f_r - c} ({x_r=}, {c=})"

    # Initially I was using Newton-Raphson instead, but since the equation has two
    # roots, there is a danger of finding the wrong solution. Bisecting can avoid
    # that by initializing properly.
    x = bisect(lambda x: f(x), c, x_l, x_r)

    return expspace(value_from, value_upto, n, grow_factor=x)


Func = Callable[[float], float | None]


def bisect(
    f: Func, target: float, x_l: float, x_r: float, tol: float = 1e-8, max_iter: int = 10000
) -> float:

    for _ in range(max_iter):
        x_mid = (x_l + x_r) * 0.5
        f_x_mid = f(x_mid)

        if f_x_mid is None:
            raise RuntimeError(f"f has overflowed for {x_mid=}")

        delta = f_x_mid - target

        if abs(delta) < tol:
            return x_mid

        if delta < 0.0:
            x_l = x_mid
        else:
            x_r = x_mid

    raise ValueError("Maximum iterations exceeded. No solution found.")
