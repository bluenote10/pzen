import numpy as np

from .signal_utils import Audio, Seconds, Time, in_samples

DEFAULT_TIME = Seconds(1.0)


class AudioGenerator:
    def __init__(self, sr: int):
        self.sr = sr

    def in_samples(self, t: Time) -> int:
        return in_samples(t, self.sr)

    def empty(self) -> Audio:
        return Audio.empty(self.sr)

    def silence(self, t: Time = DEFAULT_TIME) -> Audio:
        return Audio.silence(t, self.sr)

    def sine(self, freq: float = 440.0, t: Time = DEFAULT_TIME, phase_offset: float = 0.0) -> Audio:
        """
        Generate perfect sine from a frequency.
        """
        n = self.in_samples(t)
        ts = np.arange(n) / self.sr
        x = np.sin(phase_offset + 2.0 * np.pi * freq * ts)
        return Audio(x=x, sr=self.sr)
