from pzen.signal_gen import SignalGenerator


def test_audio_generator__consistent_dtypes():
    gen = SignalGenerator(sr=22050)
    dtypes = [
        gen.empty().x.dtype,
        gen.silence().x.dtype,
        gen.sine().x.dtype,
    ]
    assert len(set(dtypes)) == 1
