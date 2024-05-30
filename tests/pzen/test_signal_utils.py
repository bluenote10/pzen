from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from pzen.audiofile_utils import soundfile_write
from pzen.signal_utils import (
    Samples,
    Seconds,
    Signal,
    SignalGenerator,
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


def test_signal__add():
    a = Signal(x=np.array([1.0, 0.0, 0.0]), sr=1)
    b = Signal(x=np.array([0.0, 1.0, 0.0]), sr=1)
    npt.assert_allclose((a + b).x, [1.0, 1.0, 0.0])


def test_signal__pad(sr: int):
    x = Signal.empty(sr)
    assert len(x) == 0
    assert x.pad().len() == 0
    assert x.pad(pad_l=Samples(10)).len() == 10
    assert x.pad(pad_r=Samples(20)).len() == 20
    assert x.pad(pad_l=Samples(10), pad_r=Samples(20)).len() == 30

    x = Signal.zeros(Seconds(1.0), sr)
    assert len(x) == sr
    assert x.pad().len() == sr
    assert x.pad(pad_l=Samples(10)).len() == sr + 10
    assert x.pad(pad_r=Samples(20)).len() == sr + 20
    assert x.pad(pad_l=Samples(10), pad_r=Samples(20)).len() == sr + 30


def test_signal__mix_at():
    a = Signal(x=np.array([1.0, 2.0, 3.0]), sr=1)
    b = Signal(x=np.array([1.0]), sr=1)

    npt.assert_allclose(a.mix_at(Samples(0), b).x, [2.0, 2.0, 3.0])
    npt.assert_allclose(a.mix_at(Samples(1), b).x, [1.0, 3.0, 3.0])
    npt.assert_allclose(a.mix_at(Samples(2), b).x, [1.0, 2.0, 4.0])
    npt.assert_allclose(a.mix_at(Samples(3), b).x, [1.0, 2.0, 3.0, 1.0])
    npt.assert_allclose(a.mix_at(Samples(4), b).x, [1.0, 2.0, 3.0, 0.0, 1.0])

    npt.assert_allclose(a.x, [1.0, 2.0, 3.0])

    a.mix_at(Samples(2), b, allow_extend=False)
    with pytest.raises(ValueError):
        a.mix_at(Samples(3), b, allow_extend=False)

    b = Signal(x=np.array([1.0, 1.0]), sr=1)

    npt.assert_allclose(a.mix_at(Samples(0), b).x, [2.0, 3.0, 3.0])
    npt.assert_allclose(a.mix_at(Samples(1), b).x, [1.0, 3.0, 4.0])
    npt.assert_allclose(a.mix_at(Samples(2), b).x, [1.0, 2.0, 4.0, 1.0])
    npt.assert_allclose(a.mix_at(Samples(3), b).x, [1.0, 2.0, 3.0, 1.0, 1.0])
    npt.assert_allclose(a.mix_at(Samples(4), b).x, [1.0, 2.0, 3.0, 0.0, 1.0, 1.0])


# -----------------------------------------------------------------------------
# Generator API
# -----------------------------------------------------------------------------


def test_signal_generator__consistent_dtypes():
    gen = SignalGenerator(sr=22050)
    dtypes = [
        gen.empty().x.dtype,
        gen.silence().x.dtype,
        gen.sine().x.dtype,
    ]
    assert len(set(dtypes)) == 1


def test_signal_generator__envelope_ramped():
    gen = SignalGenerator(sr=22050)
    npt.assert_allclose(
        gen.envelope_ramped(Samples(8), Samples(0)).x,
        [1.0] * 8,
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(8), Samples(1)).x,
        [1.0] * 8,
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(8), Samples(2)).x,
        [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(8), Samples(3)).x,
        [1 / 3, 2 / 3, 1.0, 1.0, 1.0, 1.0, 2 / 3, 1 / 3],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(8), Samples(4)).x,
        [0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(8), Samples(5)).x,
        [0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2],
    )

    npt.assert_allclose(
        gen.envelope_ramped(Samples(8), Samples(3), Samples(4)).x,
        [1 / 3, 2 / 3, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(7), Samples(3), Samples(4)).x,
        [1 / 3, 2 / 3, 1.0, 1.0, 0.75, 0.5, 0.25],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(6), Samples(3), Samples(4)).x,
        [1 / 3, 2 / 3, 1.0, 0.75, 0.5, 0.25],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(5), Samples(3), Samples(4)).x,
        [1 / 3, 2 / 3, 0.75, 0.5, 0.25],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(4), Samples(3), Samples(4)).x,
        [1 / 3, 2 / 3, 0.5, 0.25],
    )

    npt.assert_allclose(
        gen.envelope_ramped(Samples(1), Samples(5), Samples(10)).x,
        [0.1],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(2), Samples(5), Samples(10)).x,
        [0.2, 0.1],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(3), Samples(5), Samples(10)).x,
        [0.2, 0.2, 0.1],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(4), Samples(5), Samples(10)).x,
        [0.2, 0.3, 0.2, 0.1],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(5), Samples(5), Samples(10)).x,
        [0.2, 0.4, 0.3, 0.2, 0.1],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(6), Samples(5), Samples(10)).x,
        [0.2, 0.4, 0.4, 0.3, 0.2, 0.1],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(7), Samples(5), Samples(10)).x,
        [0.2, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(8), Samples(5), Samples(10)).x,
        [0.2, 0.4, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(9), Samples(5), Samples(10)).x,
        [0.2, 0.4, 0.6, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )
    npt.assert_allclose(
        gen.envelope_ramped(Samples(10), Samples(5), Samples(10)).x,
        [0.2, 0.4, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )


def test_signal_generator__audio_examples():
    out_dir = Path("/tmp/signal_generator")
    gen = SignalGenerator(sr=22050)

    sine_enveloped = gen.sine(440).scale(0.5).envelope_ramped(Seconds(0.2))
    soundfile_write(out_dir / "sine_enveloped.wav", sine_enveloped)
