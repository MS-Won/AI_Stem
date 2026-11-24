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
