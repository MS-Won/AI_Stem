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


def band_rms(x: np.ndarray, start_s: float, end_s: float, freq_hz: float, width_hz: float = 100) -> float:
    """[start_s, end_s) 프레임에서 freq_hz 주변 대역의 STFT 크기 RMS."""
    from mrengine.spectral import N_FFT, frame_times, stft

    mag = np.abs(stft(x))
    t = frame_times(mag.shape[-1])
    f = np.fft.rfftfreq(N_FFT, 1 / SR)
    fsel = (f >= freq_hz - width_hz / 2) & (f <= freq_hz + width_hz / 2)
    tsel = (t >= start_s) & (t < end_s)
    return rms(mag[:, fsel][:, :, tsel])
