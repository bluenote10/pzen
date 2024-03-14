import numpy as np
import numpy.testing as npt

from pzen.soundfile_utils import normalize, normalize_min_max


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
