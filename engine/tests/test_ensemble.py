import numpy as np
from conftest import SR, rms

from mrengine.ensemble import ensemble
from mrengine.spectral import HOP, N_FFT, frame_times, istft, stft


def test_stft_shape_and_dtype(tone):
    x = tone(440, 1.0, 0.5)
    s = stft(x)
    assert s.dtype == np.complex64
    assert s.shape[:2] == (1, N_FFT // 2 + 1)


def test_stft_roundtrip(tone):
    x = tone(440, 1.0, 0.5)
    y = istft(stft(x), x.shape[1])
    assert y.dtype == np.float32
    assert np.allclose(y, x, atol=1e-4)


def test_frame_times_match_peak_location(tone):
    # 1.5s 지점에서 시작하는 버스트의 에너지가 해당 프레임 시각에 나타나야 한다
    x = np.zeros((1, 3 * SR), dtype=np.float32)
    x[:, int(1.5 * SR):int(1.5 * SR) + HOP] = tone(1000, 1.0, 0.5)[:, :HOP]
    s = stft(x)
    t = frame_times(s.shape[2])
    energy = np.abs(s[0]).sum(axis=0)
    assert abs(t[np.argmax(energy)] - 1.5) < 2 * N_FFT / SR
    assert t[0] == 0.0 and np.isclose(t[1] - t[0], HOP / SR)


def test_ensemble_single_input_is_identity(tone):
    x = tone(440, 1.0, 0.5)
    assert np.allclose(ensemble([x], x.shape[1]), x, atol=1e-4)


def test_ensemble_aligns_different_lengths(tone):
    a = tone(440, 1.0, 0.5)
    b = a[:, :-37]
    assert ensemble([a, b], a.shape[1], "avg").shape == a.shape


def test_min_mag_picks_quieter_model(tone):
    a = tone(440, 1.0, 0.5)
    b = tone(440, 1.0, 0.1)
    out = ensemble([a, b], a.shape[1], "min_mag")
    assert np.isclose(rms(out), rms(b), rtol=0.05)


def test_max_mag_picks_louder_model(tone):
    a = tone(440, 1.0, 0.5)
    b = tone(440, 1.0, 0.1)
    out = ensemble([a, b], a.shape[1], "max_mag")
    assert np.isclose(rms(out), rms(a), rtol=0.05)


def test_avg_is_between_models(tone):
    a = tone(440, 1.0, 0.5)
    b = tone(440, 1.0, 0.1)
    out = ensemble([a, b], a.shape[1], "avg")
    assert np.isclose(rms(out), rms(0.5 * (a + b)), rtol=0.02)
