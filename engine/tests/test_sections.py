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
    # T9에서 평활 길이를 늘려(활성 1.5초, choir 4초) 버스트를 2.5초로 키웠다.
    v, mix = _burst_song(tone, band_noise, 2.5)
    assert detect_vocal_sections(v, mix, min_len_s=3.0) == []
    kept = detect_vocal_sections(v, mix, min_len_s=2.0)
    assert len(kept) == 1 and kept[0].kind == "choir"


def test_mostly_choir_song_detects_choir(tone, band_noise):
    inst = tone(200, 11.0, 0.1)
    v = np.zeros_like(inst)
    v[:, SR : 9 * SR] = band_noise(300, 6000, 8.0, 0.5)
    v[:, 9 * SR :] = tone(440, 2.0, 0.2) + tone(880, 2.0, 0.15) + tone(1320, 2.0, 0.1)
    secs = detect_vocal_sections(v, inst + v)
    covered = sum(min(s.end_s, 9) - max(s.start_s, 1) for s in secs if s.kind == "choir" and s.end_s > 1 and s.start_s < 9)
    assert covered >= 7.0


def test_digital_silence_next_to_short_section_makes_no_choir(tone):
    """완전 무음 프레임은 평탄도가 1로 계산돼 'choir 같은' 값이 나온다. 활성 프레임만 choir로 판정하지
    않으면, 무음 바로 앞의 lead 구간이 choir 평활(4초)에 끌려 choir로 바뀐다 (T9 리뷰: 버스트는 평활을
    통과하는 2.5초)."""
    inst = tone(200, 8.0, 0.1)
    v = np.zeros_like(inst)
    a, b = 2 * SR, int(4.5 * SR)
    v[:, a:b] = tone(440, 2.5, 0.3) + tone(880, 2.5, 0.15)  # 좁은 대역 = lead
    mix = inst + v
    mix[:, b:] = 0  # 버스트 뒤는 완전 무음
    secs = detect_vocal_sections(v, mix, min_len_s=0.5)
    assert any(s.kind == "lead" and s.start_s < 3.0 and s.end_s > 4.0 for s in secs)  # 버스트는 lead로 검출
    assert [s for s in secs if s.kind == "choir"] == []
    assert [s for s in secs if s.start_s > 4.5] == []


def test_choir_with_brief_narrow_moments_stays_one_section(tone, band_noise):
    """실제 곡에서는 프레임 폭 판정이 흔들려 choir/lead가 1초 단위로 번갈아 나왔다(T9).
    넓은 대역 0.8초 + 좁은 음 0.4초가 반복되는 8초 합창은 choir 구간 하나여야 한다."""
    inst = tone(200, 12.0, 0.1)
    v = np.zeros_like(inst)
    t = 2.0
    while t < 10.0 - 1e-9:
        a = int(round(t * SR))
        v[:, a : a + int(0.8 * SR)] = band_noise(300, 6000, 0.8, 0.5, seed=int(t * 10))
        b = a + int(0.8 * SR)
        v[:, b : b + int(0.4 * SR)] = tone(440, 0.4, 0.4)
        t += 1.2
    secs = detect_vocal_sections(v, inst + v)
    choir = [s for s in secs if s.kind == "choir"]
    assert len(choir) == 1
    assert choir[0].start_s < 3.0 and choir[0].end_s > 9.0
