import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d
from conftest import band_rms

from mrengine.restore import restore_choir, restore_choir_bins
from mrengine.sections import Section
from mrengine.spectral import frame_times, stft

SR = 44100
SECS = [Section(3.0, 6.0, "choir")]


@pytest.fixture
def song(tone, band_noise):
    """악기 = 1kHz+200Hz 톤(전 구간), 합창 = 300Hz~6kHz 노이즈(3~6초).
    inst 추정치는 3~6초에 1kHz 성분을 0.1배로 깎은 것(spectral hole)."""
    k1 = tone(1000, 9.0, 0.3)
    instrument = k1 + tone(200, 9.0, 0.2)
    vocals = np.zeros_like(instrument)
    vocals[:, 3 * SR : 6 * SR] = band_noise(300, 6000, 3.0, 0.4)
    mix = instrument + vocals
    inst = instrument.copy()
    # 경계 클릭이 생기지 않도록 0.2초 페이드로 구멍을 만든다.
    gain = np.zeros(k1.shape[-1])
    gain[3 * SR : 6 * SR] = 0.9
    gain = uniform_filter1d(gain, int(0.2 * SR))
    inst -= (gain * k1).astype(np.float32)
    return mix, inst, vocals, instrument


def test_strength_zero_is_identity(song):
    mix, inst, vocals, _ = song
    out, _ = restore_choir(mix, inst, vocals, SECS, strength=0.0)
    assert np.allclose(out, inst, atol=1e-4)


def test_fills_hole_in_choir_section(song):
    mix, inst, vocals, instrument = song
    out, _ = restore_choir(mix, inst, vocals, SECS, strength=1.0)
    before = band_rms(inst, 3.5, 5.5, 1000)
    after = band_rms(out, 3.5, 5.5, 1000)
    full = band_rms(instrument, 3.5, 5.5, 1000)
    assert before < 0.2 * full
    assert after >= 0.7 * full


def test_does_not_touch_outside_sections(song):
    mix, inst, vocals, _ = song
    out, _ = restore_choir(mix, inst, vocals, SECS, strength=1.0)
    assert np.allclose(out[:, : int(2.9 * SR)], inst[:, : int(2.9 * SR)], atol=1e-4)
    assert np.allclose(out[:, int(6.2 * SR) :], inst[:, int(6.2 * SR) :], atol=1e-4)


def test_bins_never_exceed_mix():
    rng = np.random.default_rng(1)
    shape = (2, 300, 40)
    c = lambda: (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    M, I, V = c(), c(), c()
    ref = rng.uniform(0.2, 1.5, shape[1]).astype(np.float32)
    s = rng.uniform(0, 1, shape[2]).astype(np.float32)
    out = restore_choir_bins(M, I, V, ref, s)
    assert np.all(np.abs(out) <= np.abs(M) + 1e-7 * np.abs(M).max())


def _excess(x, mix_mag):
    """믹스 크기를 넘은 양의 에너지: 모든 칸의 max(|stft(x)| - |stft(mix)|, 0)^2 합."""
    return float(np.sum(np.maximum(np.abs(stft(x)) - mix_mag, 0.0).astype(np.float64) ** 2))


def _assert_overshoot_not_worse(out, inst, mix):
    """R9: 재합성 후 초과 에너지가 입력 inst의 초과 에너지보다 믹스 에너지의 1e-4 이상 늘지 않는다."""
    mix_mag = np.abs(stft(mix))
    e_out, e_in = _excess(out, mix_mag), _excess(inst, mix_mag)
    print(f"excess out={e_out:.4g} inst={e_in:.4g} mix_energy={np.sum(mix_mag.astype(np.float64) ** 2):.4g}")
    assert e_out <= e_in + 1e-4 * np.sum(mix_mag.astype(np.float64) ** 2)


def test_resynthesis_never_worsens_overshoot(song):
    mix, inst, vocals, _ = song
    out, _ = restore_choir(mix, inst, vocals, SECS, strength=1.0)
    _assert_overshoot_not_worse(out, inst, mix)


def test_resynthesis_never_worsens_overshoot_dense_noise(tone, band_noise):
    instrument = tone(1000, 9.0, 0.2) + band_noise(100, 10000, 9.0, 0.1, seed=3)
    vocals = np.zeros_like(instrument)
    vocals[:, 3 * SR : 6 * SR] = band_noise(300, 6000, 3.0, 0.3, seed=4)
    mix = instrument + vocals
    inst = instrument * 0.5
    out, _ = restore_choir(mix, inst, vocals, SECS, strength=1.0)
    _assert_overshoot_not_worse(out, inst, mix)


def test_no_choir_sections_returns_input(song):
    mix, inst, vocals, _ = song
    out, amount = restore_choir(mix, inst, vocals, [Section(3.0, 6.0, "lead")], strength=1.0)
    assert np.array_equal(out, inst)
    assert not amount.any()
    out, _ = restore_choir(mix, inst, vocals, [], strength=1.0)
    assert np.array_equal(out, inst)


def test_restore_amount_map_is_positive_only_in_choir(song):
    mix, inst, vocals, _ = song
    out, amount = restore_choir(mix, inst, vocals, SECS, strength=1.0)
    t = frame_times(amount.shape[0])
    assert amount.dtype == np.float32 and amount.shape == (stft(mix).shape[-1],)
    inside = (t >= 3.1) & (t < 5.9)
    assert np.all(amount[inside] > 0)
    assert np.all(amount[(t < 3.0) | (t >= 6.0)] == 0)
    assert out.shape == inst.shape and out.dtype == np.float32
