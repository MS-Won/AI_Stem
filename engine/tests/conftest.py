import numpy as np
import pytest

SR = 44100


@pytest.fixture
def tone():
    def make(freq_hz: float, seconds: float, amp: float = 0.5) -> np.ndarray:
        t = np.arange(int(round(seconds * SR))) / SR
        return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)[None, :]

    return make


@pytest.fixture
def band_noise():
    def make(lo_hz: float, hi_hz: float, seconds: float, amp: float = 0.5, seed: int = 0) -> np.ndarray:
        n = int(round(seconds * SR))
        rng = np.random.default_rng(seed)
        spec = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / SR)
        spec[(f < lo_hz) | (f > hi_hz)] = 0
        x = np.fft.irfft(spec, n)
        x = x / (np.max(np.abs(x)) + 1e-12) * amp
        return x.astype(np.float32)[None, :]

    return make


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
