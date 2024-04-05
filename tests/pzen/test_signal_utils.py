import numpy as np
import numpy.testing as npt

from pzen.signal_utils import normalize, normalize_min_max, pad_to_blocksize


def test_normalize():
    npt.assert_allclose(
        normalize(np.array([0.5, 0.0, -0.25])),
        np.array([1.0, 0.0, -0.5]),
    )
    npt.assert_allclose(
        normalize(np.array([0.5, 0.5, 0.5])),
        np.array([1.0, 1.0, 1.0]),
    )
    npt.assert_allclose(
        normalize(np.array([0.0, 0.0, 0.0])),
        np.array([0.0, 0.0, 0.0]),
    )


def test_normalize_min_max():
    npt.assert_allclose(
        normalize_min_max(np.array([0.5, 0.0, -0.25])),
        np.array([1.0, -1 / 3, -1.0]),
    )
    npt.assert_allclose(
        normalize_min_max(np.array([0.5, 0.5, 0.5])),
        np.array([0.5, 0.5, 0.5]),
    )
    npt.assert_allclose(
        normalize_min_max(np.array([0.0, 0.0, 0.0])),
        np.array([0.0, 0.0, 0.0]),
    )


def test_pad_to_blocksize():
    x, n_pad = pad_to_blocksize(np.array([]), 3)
    assert n_pad == 0
    npt.assert_allclose(x, np.array([]))

    x, n_pad = pad_to_blocksize(np.array([1.0]), 3)
    assert n_pad == 2
    npt.assert_allclose(x, np.array([1.0, 0.0, 0.0]))

    x, n_pad = pad_to_blocksize(np.array([1.0, 2.0]), 3)
    assert n_pad == 1
    npt.assert_allclose(x, np.array([1.0, 2.0, 0.0]))

    x, n_pad = pad_to_blocksize(np.array([1.0, 2.0, 3.0]), 3)
    assert n_pad == 0
    npt.assert_allclose(x, np.array([1.0, 2.0, 3.0]))
