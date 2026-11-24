import numpy as np
from conftest import SR, rms_db

from mrengine.loudness import finalize, restore_section_loudness
from mrengine.sections import Section


def _dropped(tone, drop_db):
    """mix는 일정, inst는 3~6초만 drop_db 만큼 작은 10초 신호."""
    mix = tone(1000, 10.0, 0.5)
    inst = tone(1000, 10.0, 0.25)
    inst[:, 3 * SR : 6 * SR] *= 10 ** (drop_db / 20)
    return inst, mix


def test_raises_dropped_section(tone):
    inst, mix = _dropped(tone, -6.0)
    out, gains = restore_section_loudness(inst, mix, [Section(3, 6, "choir")])
    assert abs(rms_db(out, 3.3, 5.7) - rms_db(out, 0.5, 2.5)) < 1.0
    assert 5.0 < gains[0] <= 6.0
    assert out.shape == inst.shape and out.dtype == np.float32
    np.testing.assert_allclose(out[:, : 2 * SR], inst[:, : 2 * SR])  # 램프 밖은 불변


def test_gain_capped(tone):
    inst, mix = _dropped(tone, -15.0)
    out, gains = restore_section_loudness(inst, mix, [Section(3, 6, "choir")])
    assert gains[0] == 6.0


def test_lead_sections_untouched(tone):
    inst, mix = _dropped(tone, -6.0)
    out, gains = restore_section_loudness(inst, mix, [Section(3, 6, "lead")])
    assert gains == [0.0]
    np.testing.assert_array_equal(out, inst)


def test_never_cuts(tone):
    inst = tone(1000, 10.0, 0.25)
    mix = tone(1000, 10.0, 0.5)
    mix[:, 3 * SR : 6 * SR] *= 0.25  # 구간 안 비율이 기준보다 높음
    _, gains = restore_section_loudness(inst, mix, [Section(3, 6, "choir")])
    assert gains == [0.0]


def test_no_reference_gives_zero_gain(tone):
    inst, mix = _dropped(tone, -6.0)
    _, gains = restore_section_loudness(inst[:, : 6 * SR], mix[:, : 6 * SR], [Section(0, 6, "choir")])
    assert gains == [0.0]


def test_finalize_peak_never_exceeds_minus_1dbfs(tone):
    out = finalize(tone(440, 1.0, 1.4))
    assert 20 * np.log10(np.max(np.abs(out))) <= -1.0 + 1e-3
    assert out.dtype == np.float32


def test_finalize_leaves_quiet_audio(tone):
    x = tone(440, 1.0, 0.5)  # 피크 약 -6dB
    np.testing.assert_array_equal(finalize(x), x)


def test_finalize_silence():
    out = finalize(np.zeros((1, 100), dtype=np.float32))
    assert not np.any(out)
