import numpy as np

from .signal_utils import Seconds, Signal, Time, in_samples

DEFAULT_TIME = Seconds(1.0)


class SignalGenerator:
    def __init__(self, sr: int):
        self.sr = sr

    def _in_samples(self, t: Time) -> int:
        return in_samples(t, self.sr)

    def _make_signal(self, x: np.ndarray) -> Signal:
        return Signal(x=x, sr=self.sr)

    # General purpose generators

    def ramp(self, t: Time, start: float = 0.0, end: float = 1.0) -> Signal:
        n = self._in_samples(t)
        return self._make_signal(np.arange(start, end, n))

    # Possible extension would be to have `multi_ramp` that takes a vararg of
    # tuples of (delta: Time, value: float), allowing to chain linear ramps

    # Envelope generators

    def envelope_ramped(self, t: Time, t_l: Time, t_r: Time | None = None) -> Signal:
        """
        Returns an envelope that ramps up from 0 -> 1 and back from 1 -> 0.
        """
        if t_r is None:
            t_r = t_l
        n = self._in_samples(t)
        n_l = self._in_samples(t_l)
        n_r = self._in_samples(t_r)
        x = np.ones(n)
        x[0:n_l] = np.linspace(0.0, 1.0, n_l)

    # Audio-like generators

    def empty(self) -> Signal:
        return Signal.empty(self.sr)

    def silence(self, t: Time = DEFAULT_TIME) -> Signal:
        return Signal.zeros(t, self.sr)

    def sine(
        self, freq: float = 440.0, t: Time = DEFAULT_TIME, phase_offset: float = 0.0
    ) -> Signal:
        """
        Generate perfect sine from a frequency.
        """
        n = self._in_samples(t)
        ts = np.arange(n) / self.sr
        x = np.sin(phase_offset + 2.0 * np.pi * freq * ts)
        return self._make_signal(x)
