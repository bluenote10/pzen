import numpy as np
import numpy.testing as npt
import pytest

from pzen.signal_utils import (
    Audio,
    NumSamples,
    Seconds,
    normalize,
    normalize_min_max,
    pad_to_multiple_of,
)

# -----------------------------------------------------------------------------
# Low-level API
# -----------------------------------------------------------------------------


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
    x = pad_to_multiple_of(np.array([]), 3)
    npt.assert_allclose(x, np.array([]))

    x = pad_to_multiple_of(np.array([1.0]), 3)
    npt.assert_allclose(x, np.array([1.0, 0.0, 0.0]))

    x = pad_to_multiple_of(np.array([1.0, 2.0]), 3)
    npt.assert_allclose(x, np.array([1.0, 2.0, 0.0]))

    x = pad_to_multiple_of(np.array([1.0, 2.0, 3.0]), 3)
    npt.assert_allclose(x, np.array([1.0, 2.0, 3.0]))


# -----------------------------------------------------------------------------
# High-level API
# -----------------------------------------------------------------------------


@pytest.fixture(params=[22050, 44100])
def sr(request: pytest.FixtureRequest) -> int:
    sr_value = request.param
    assert isinstance(sr_value, int)
    return sr_value


def test_audio__add():
    a = Audio(x=np.array([1.0, 0.0, 0.0]), sr=1)
    b = Audio(x=np.array([0.0, 1.0, 0.0]), sr=1)
    npt.assert_allclose((a + b).x, [1.0, 1.0, 0.0])


def test_audio__pad(sr: int):
    audio = Audio.empty(sr)
    assert len(audio) == 0
    assert audio.pad().len() == 0
    assert audio.pad(pad_l=NumSamples(10)).len() == 10
    assert audio.pad(pad_r=NumSamples(20)).len() == 20
    assert audio.pad(pad_l=NumSamples(10), pad_r=NumSamples(20)).len() == 30

    audio = Audio.silence(Seconds(1.0), sr)
    assert len(audio) == sr
    assert audio.pad().len() == sr
    assert audio.pad(pad_l=NumSamples(10)).len() == sr + 10
    assert audio.pad(pad_r=NumSamples(20)).len() == sr + 20
    assert audio.pad(pad_l=NumSamples(10), pad_r=NumSamples(20)).len() == sr + 30


def test_audio__mix_at():
    a = Audio(x=np.array([1.0, 2.0, 3.0]), sr=1)
    b = Audio(x=np.array([1.0]), sr=1)

    npt.assert_allclose(a.mix_at(NumSamples(0), b).x, [2.0, 2.0, 3.0])
    npt.assert_allclose(a.mix_at(NumSamples(1), b).x, [1.0, 3.0, 3.0])
    npt.assert_allclose(a.mix_at(NumSamples(2), b).x, [1.0, 2.0, 4.0])
    npt.assert_allclose(a.mix_at(NumSamples(3), b).x, [1.0, 2.0, 3.0, 1.0])
    npt.assert_allclose(a.mix_at(NumSamples(4), b).x, [1.0, 2.0, 3.0, 0.0, 1.0])

    npt.assert_allclose(a.x, [1.0, 2.0, 3.0])

    b = Audio(x=np.array([1.0, 1.0]), sr=1)

    npt.assert_allclose(a.mix_at(NumSamples(0), b).x, [2.0, 3.0, 3.0])
    npt.assert_allclose(a.mix_at(NumSamples(1), b).x, [1.0, 3.0, 4.0])
    npt.assert_allclose(a.mix_at(NumSamples(2), b).x, [1.0, 2.0, 4.0, 1.0])
    npt.assert_allclose(a.mix_at(NumSamples(3), b).x, [1.0, 2.0, 3.0, 1.0, 1.0])
    npt.assert_allclose(a.mix_at(NumSamples(4), b).x, [1.0, 2.0, 3.0, 0.0, 1.0, 1.0])
