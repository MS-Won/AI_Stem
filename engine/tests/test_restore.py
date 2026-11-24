import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d
from conftest import band_rms

from mrengine.restore import restore_choir
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
    assert np.allclose(out[:, : int(2.8 * SR)], inst[:, : int(2.8 * SR)], atol=1e-4)
    assert np.allclose(out[:, int(6.2 * SR) :], inst[:, int(6.2 * SR) :], atol=1e-4)


def test_never_exceeds_mix_magnitude(song):
    mix, inst, vocals, _ = song
    out, _ = restore_choir(mix, inst, vocals, SECS, strength=1.0)
    over = np.abs(stft(out)) - np.abs(stft(mix))
    t = frame_times(over.shape[-1])
    # 보장 대상은 복구가 손댄 choir 프레임. 구간 밖은 입력 inst 그대로라 입력의 오차까지 보장하지 않는다.
    assert np.all(over[:, :, (t >= 3.0) & (t < 6.0)] <= 1e-3)


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
