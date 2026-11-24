import numpy as np
import pytest

from mrengine.sections import Section, detect_vocal_sections, frames_mask
from mrengine.spectral import frame_times, stft

SR = 44100


@pytest.fixture
def synthetic_song(tone, band_noise):
    """0~3초 악기, 3~6초 악기+넓은 노이즈(choir), 6~9초 악기+좁은 톤 묶음(lead)."""
    n = 3 * SR
    inst = tone(200, 9.0, 0.1)
    v = np.zeros_like(inst)
    v[:, n : 2 * n] = band_noise(300, 6000, 3.0, 0.5)
    lead = tone(440, 3.0, 0.2) + tone(880, 3.0, 0.15) + tone(1320, 3.0, 0.1)
    v[:, 2 * n :] = lead
    return v, inst + v


def test_detects_choir_section(synthetic_song):
    v, mix = synthetic_song
    choir = [s for s in detect_vocal_sections(v, mix) if s.kind == "choir"]
    assert len(choir) == 1
    assert abs(choir[0].start_s - 3) < 0.3 and abs(choir[0].end_s - 6) < 0.3


def test_detects_lead_section(synthetic_song):
    v, mix = synthetic_song
    lead = [s for s in detect_vocal_sections(v, mix) if s.kind == "lead"]
    assert len(lead) == 1
    assert abs(lead[0].start_s - 6) < 0.3 and abs(lead[0].end_s - 9) < 0.3


def test_no_vocals_gives_no_sections(tone):
    x = tone(200, 5.0, 0.5)
    assert detect_vocal_sections(np.zeros_like(x), x) == []


def test_short_blips_are_dropped(tone, band_noise):
    inst = tone(200, 5.0, 0.1)
    v = np.zeros_like(inst)
    v[:, 2 * SR : 2 * SR + int(0.3 * SR)] = band_noise(300, 6000, 0.3, 0.5)
    assert detect_vocal_sections(v, inst + v) == []


def test_frames_mask_uses_frame_times():
    secs = [Section(1.0, 2.0, "choir"), Section(3.0, 4.0, "lead")]
    n = 200
    t = frame_times(n)
    m = frames_mask(secs, n, {"choir"})
    assert m.dtype == bool and np.array_equal(m, (t >= 1.0) & (t < 2.0))
    both = frames_mask(secs, n, {"choir", "lead"})
    assert np.array_equal(both, ((t >= 1) & (t < 2)) | ((t >= 3) & (t < 4)))
    assert not frames_mask([], n, {"choir"}).any()


def test_section_to_dict():
    assert Section(1.0, 2.5, "lead").to_dict() == {"start_s": 1.0, "end_s": 2.5, "kind": "lead"}


def test_synthetic_stft_shape_sanity(synthetic_song):
    # 합성 신호가 두 구간 모두에서 v_ratio 임계 위에 있는지 (테스트 설계 확인)
    v, mix = synthetic_song
    r = (np.abs(stft(v)) ** 2).sum((0, 1)) / (np.abs(stft(mix)) ** 2).sum((0, 1))
    t = frame_times(len(r))
    assert r[(t > 3.5) & (t < 5.5)].min() > 0.25 and r[(t > 6.5) & (t < 8.5)].min() > 0.25


def _burst_song(tone, band_noise, seconds):
    inst = tone(200, 6.0, 0.1)
    v = np.zeros_like(inst)
    n = int(round(seconds * SR))
    v[:, 2 * SR : 2 * SR + n] = band_noise(300, 6000, seconds, 0.5)
    return v, inst + v


def test_min_len_drops_burst_that_survives_smoothing(tone, band_noise):
    v, mix = _burst_song(tone, band_noise, 0.7)
    assert detect_vocal_sections(v, mix, min_len_s=1.0) == []
    kept = detect_vocal_sections(v, mix, min_len_s=0.5)
    assert len(kept) == 1 and kept[0].kind == "choir"


def test_mostly_choir_song_detects_choir(tone, band_noise):
    inst = tone(200, 11.0, 0.1)
    v = np.zeros_like(inst)
    v[:, SR : 9 * SR] = band_noise(300, 6000, 8.0, 0.5)
    v[:, 9 * SR :] = tone(440, 2.0, 0.2) + tone(880, 2.0, 0.15) + tone(1320, 2.0, 0.1)
    secs = detect_vocal_sections(v, inst + v)
    covered = sum(min(s.end_s, 9) - max(s.start_s, 1) for s in secs if s.kind == "choir" and s.end_s > 1 and s.start_s < 9)
    assert covered >= 7.0


def test_digital_silence_next_to_short_section_makes_no_choir(tone, band_noise):
    v, mix = _burst_song(tone, band_noise, 0.7)
    mix[:, 3 * SR :] = 0  # 이후는 완전 무음
    assert [s for s in detect_vocal_sections(v, mix, min_len_s=0.5) if s.start_s > 3.0] == []
