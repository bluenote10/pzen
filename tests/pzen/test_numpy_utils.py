import numpy as np
import numpy.testing as npt
import pytest

from pzen.numpy_utils import expspace, expspace_delta, find_peaks


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


def test_expspace():
    npt.assert_allclose(
        expspace(0, 1, n=2, grow_factor=2.0),
        np.array([0.0, 1.0]),
    )
    npt.assert_allclose(
        expspace(5, 8, n=2, grow_factor=2.0),
        np.array([5.0, 8.0]),
    )
    npt.assert_allclose(
        expspace(-5, -8, n=2, grow_factor=2.0),
        np.array([-5.0, -8.0]),
    )

    npt.assert_allclose(
        expspace(0, 1, n=3, grow_factor=2.0),
        np.array([0.0, 1 / 3, 1.0]),
    )
    npt.assert_allclose(
        expspace(0, 1, n=3, grow_factor=0.5),
        np.array([0.0, 2 / 3, 1.0]),
    )

    npt.assert_allclose(
        expspace(5, 8, n=3, grow_factor=2.0),
        np.array([5.0, 6.0, 8.0]),
    )
    npt.assert_allclose(
        expspace(5, 8, n=3, grow_factor=0.5),
        np.array([5.0, 7.0, 8.0]),
    )

    npt.assert_allclose(
        expspace(-5, -8, n=3, grow_factor=2.0),
        np.array([-5.0, -6.0, -8.0]),
    )
    npt.assert_allclose(
        expspace(-5, -8, n=3, grow_factor=0.5),
        np.array([-5.0, -7.0, -8.0]),
    )


def test_expspace_delta__n_2():
    npt.assert_allclose(
        expspace_delta(5.0, 6.0, n=2, delta=1.0),
        np.array([5.0, 6.0]),
    )
    with pytest.raises(ValueError):
        expspace_delta(5.0, 6.0, n=2, delta=2.0)


def test_expspace_delta__positive_growing():
    npt.assert_allclose(
        expspace_delta(5.0, 8.0, n=3, delta=1.0),
        np.array([5.0, 6.0, 8.0]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 6.5, n=3, delta=0.5),
        np.array([5.0, 5.5, 6.5]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 12.0, n=4, delta=1.0),
        np.array([5.0, 6.0, 8.0, 12.0]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 8.5, n=4, delta=0.5),
        np.array([5.0, 5.5, 6.5, 8.5]),
    )


def test_expspace_delta__positive_shrinking():
    npt.assert_allclose(
        expspace_delta(5.0, 6.5, n=3, delta=1.0),
        np.array([5.0, 6.0, 6.5]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 5.75, n=3, delta=0.5),
        np.array([5.0, 5.5, 5.75]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 6.75, n=4, delta=1.0),
        np.array([5.0, 6.0, 6.5, 6.75]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 5.875, n=4, delta=0.5),
        np.array([5.0, 5.5, 5.75, 5.875]),
    )


def test_expspace_delta__negative_growing():
    npt.assert_allclose(
        expspace_delta(5.0, 2.0, n=3, delta=-1.0),
        np.array([5.0, 4.0, 2.0]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 3.5, n=3, delta=-0.5),
        np.array([5.0, 4.5, 3.5]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, -2.0, n=4, delta=-1.0),
        np.array([5.0, 4.0, 2.0, -2.0]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 1.5, n=4, delta=-0.5),
        np.array([5.0, 4.5, 3.5, 1.5]),
    )


def test_expspace_delta__negative_shrinking():
    npt.assert_allclose(
        expspace_delta(5.0, 3.5, n=3, delta=-1.0),
        np.array([5.0, 4.0, 3.5]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 4.25, n=3, delta=-0.5),
        np.array([5.0, 4.5, 4.25]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 3.25, n=4, delta=-1.0),
        np.array([5.0, 4.0, 3.5, 3.25]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 4.125, n=4, delta=-0.5),
        np.array([5.0, 4.5, 4.25, 4.125]),
    )


def test_expspace_delta__neutral():
    npt.assert_allclose(
        expspace_delta(5.0, 7.0, n=3, delta=1.0),
        np.array([5.0, 6.0, 7.0]),
    )
    npt.assert_allclose(
        expspace_delta(5.0, 3.0, n=3, delta=-1.0),
        np.array([5.0, 4.0, 3.0]),
    )


@pytest.mark.parametrize("n", [3, 10, 100, 1000, 10000])
def test_expspace_delta__misc(n: int):
    result = expspace_delta(0.0, 1000.0, n=n, delta=10.0)
    npt.assert_allclose(
        result[:2],
        np.array([0.0, 10.0]),
    )
    npt.assert_allclose(
        result[-1],
        np.array(1000.0),
    )
