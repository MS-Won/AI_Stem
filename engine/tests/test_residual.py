import pytest
import numpy as np

from conftest import SR, FakeSeparator, rms
from mrengine.residual import scan_and_reduce


def _fixture(tone, band_noise):
    mr = tone(440, 6.0, amp=0.5)
    vocal = np.zeros_like(mr)
    vocal[:, 2 * SR : 3 * SR] = band_noise(300, 3000, 1.0, amp=0.5)
    mr = mr + vocal  # 2~3초에만 보컬 잔여가 섞인 MR (약 -10 ~ -20dB 수준)
    return mr.astype(np.float32), vocal


def test_finds_residual_span(tone, band_noise):
    mr, vocal = _fixture(tone, band_noise)
    fake = FakeSeparator(vocal)
    out, spans = scan_and_reduce(mr, fake, threshold_db=-30)
    assert len(spans) == 1 and abs(spans[0].start_s - 2) < 0.5
    assert abs(spans[0].end_s - 3) < 0.5
    assert fake.calls == 1
    assert set(spans[0].to_dict()) == {"start_s", "end_s", "level_db"}
    assert rms(out[:, int(2.3 * SR) : int(2.7 * SR)]) < rms(mr[:, int(2.3 * SR) : int(2.7 * SR)])


def test_reduces_only_in_span(tone, band_noise):
    mr, vocal = _fixture(tone, band_noise)
    out, spans = scan_and_reduce(mr, FakeSeparator(vocal), threshold_db=-30)
    s, e = int(spans[0].start_s * SR), int(spans[0].end_s * SR)
    assert out.shape == mr.shape and out.dtype == np.float32
    assert np.array_equal(out[:, :s], mr[:, :s])
    assert np.array_equal(out[:, e:], mr[:, e:])
    assert not np.array_equal(out[:, s:e], mr[:, s:e])


def test_clean_mr_is_unchanged(tone):
    mr = tone(440, 4.0)
    out, spans = scan_and_reduce(mr, FakeSeparator(np.zeros_like(mr)))
    assert spans == [] and np.array_equal(out, mr)


def test_silent_mr_returns_empty():
    mr = np.zeros((1, SR), dtype=np.float32)
    out, spans = scan_and_reduce(mr, FakeSeparator(mr.copy()))
    assert spans == [] and np.array_equal(out, mr)


def test_boosted_mr_gives_same_spans_and_scaled_output(tone, band_noise):
    # 음량 보정으로 피크가 1을 넘은 MR(x2.5)도 같은 구간을 찾고, 결과는 같은 배율의 결과여야 한다.
    mr, vocal = _fixture(tone, band_noise)
    k = 2.5
    out1, spans1 = scan_and_reduce(mr, FakeSeparator(vocal), threshold_db=-30)
    outk, spansk = scan_and_reduce(mr * k, FakeSeparator(vocal * k), threshold_db=-30)
    assert np.abs(mr * k).max() > 1.0
    assert [(s.start_s, s.end_s) for s in spans1] == [(s.start_s, s.end_s) for s in spansk]
    np.testing.assert_allclose(outk, out1 * k, atol=1e-4)


class _MaskLib:
    """라이브러리 흉내: 입력 피크 1 초과 시 1로 줄이고, 2~3초 구간을 보컬로 내보낸다."""

    def __init__(self, d):
        self.d = d

    def separate(self, path):
        import soundfile as sf

        x, _ = sf.read(path, dtype="float32", always_2d=True)
        if np.abs(x).max() > 1.0:
            x = x / np.abs(x).max()
        v = np.zeros_like(x)
        v[2 * SR : 3 * SR] = x[2 * SR : 3 * SR]
        for kind, y in (("Vocals", v), ("Instrumental", x - v)):
            sf.write(str(self.d / f"input_({kind})_m.wav"), y, SR, subtype="FLOAT")
        return [f"input_({kind})_m.wav" for kind in ("Vocals", "Instrumental")]


def test_backend_plus_residual_is_scale_invariant(tmp_path, tone):
    from mrengine.separate import AudioSeparatorBackend

    be = AudioSeparatorBackend("fake", tmp_path, tmp_path, _sep=_MaskLib(tmp_path))
    mr = tone(440, 6.0, amp=0.5)
    out1, spans1 = scan_and_reduce(mr, be)
    outk, spansk = scan_and_reduce(mr * 4, be)  # 피크 2.0 > 1
    assert len(spans1) == 1 and [(s.start_s, s.end_s) for s in spans1] == [(s.start_s, s.end_s) for s in spansk]
    assert spansk[0].level_db == pytest.approx(spans1[0].level_db, abs=0.01)
    np.testing.assert_allclose(outk, out1 * 4, atol=1e-4)
