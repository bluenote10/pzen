import numpy as np
import numpy.testing as npt

from pzen.numpy_utils import find_peaks


def test_find_peaks__basics():
    npt.assert_equal(
        find_peaks(np.array([1.0, 2.0, 0.2, 3.0, 2.3]), distance=1),
        np.array([1, 3]),
    )
    npt.assert_equal(
        find_peaks(np.array([1.0, 2.0, 0.2, 3.0, 2.3]), distance=2),
        np.array([3]),
    )
    npt.assert_equal(
        find_peaks(np.array([1.0, 2.0, 0.2, 3.0, 2.3]), distance=10),
        np.array([3]),
    )


def test_find_peaks__repeated_peak_value():
    npt.assert_equal(
        find_peaks(np.array([1.0, 1.0, 3.0, 3.0, 1.0, 1.0]), distance=1),
        np.array([2]),
    )


def test_find_peaks__test_case_from_issue_18495():
    # https://github.com/scipy/scipy/issues/18495
    x = np.zeros(200)
    x[100:] = np.linspace(1.0, 0.0, 100) + np.random.normal(0.0, 1e-2, size=100)

    assert len(find_peaks(x, distance=10)) == 1
