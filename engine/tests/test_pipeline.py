import json

import numpy as np
import pytest

from conftest import FakeSeparator
from mrengine.audio_io import SR, load_audio, write_wav
from mrengine.pipeline import PipelineConfig, run_pipeline, separate_all


def _song(tone, band_noise, channels=1, seconds=12.0):
    """0~4초 반주만, 4~8초 넓은 대역 보컬(합창 유사), 8~12초 좁은 보컬(lead 유사)."""
    inst = tone(220, seconds, 0.3) + tone(330, seconds, 0.2)
    voc = np.zeros_like(inst)
    n4 = 4 * SR
    voc[:, n4 : 2 * n4] = band_noise(300, 6000, 4, 0.3)
    voc[:, 2 * n4 : 3 * n4] = tone(800, 4, 0.3)
    if channels == 2:
        inst, voc = np.repeat(inst, 2, axis=0), np.repeat(voc, 2, axis=0)
    return inst + voc, voc, inst


@pytest.fixture
def song(tmp_path, tone, band_noise):
    def make(channels=1, vocals_zero=False):
        mix, voc, inst = _song(tone, band_noise, channels)
        if vocals_zero:
            mix, voc = inst.copy(), np.zeros_like(inst)
        path = tmp_path / "in.wav"
        write_wav(path, mix)
        seps = [FakeSeparator(voc, inst, name="a"), FakeSeparator(voc, inst, name="b")]
        return path, seps

    return make


def test_pipeline_writes_all_outputs(tmp_path, song):
    path, seps = song()
    out = tmp_path / "out"
    rep = run_pipeline(path, out, seps)
    for f in ["mr_baseline.wav", "mr_restored.wav", "mr_baseline.mp3", "mr_restored.mp3", "report.json"]:
        assert (out / f).exists()
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == rep


def test_pipeline_skips_mp3_when_disabled(tmp_path, song):
    path, seps = song()
    run_pipeline(path, tmp_path / "o", seps, PipelineConfig(write_mp3=False))
    assert (tmp_path / "o" / "mr_restored.wav").exists()
    assert not (tmp_path / "o" / "mr_restored.mp3").exists()


@pytest.mark.parametrize("channels", [1, 2])
def test_pipeline_preserves_length_and_channels(tmp_path, song, channels):
    path, seps = song(channels)
    n = load_audio(path).shape[1]
    run_pipeline(path, tmp_path / "o", seps, PipelineConfig(write_mp3=False))
    for f in ["mr_baseline.wav", "mr_restored.wav"]:
        a = load_audio(tmp_path / "o" / f)
        assert a.shape == (channels, n)


def test_pipeline_no_vocal_sections_returns_baseline(tmp_path, song):
    path, seps = song(vocals_zero=True)
    rep = run_pipeline(path, tmp_path / "o", seps, PipelineConfig(write_mp3=False))
    assert rep["sections"] == []
    a = load_audio(tmp_path / "o" / "mr_baseline.wav")
    b = load_audio(tmp_path / "o" / "mr_restored.wav")
    np.testing.assert_allclose(a, b, atol=1e-3)


def test_report_has_required_keys(tmp_path, song):
    path, seps = song()
    rep = run_pipeline(path, tmp_path / "o", seps, PipelineConfig(write_mp3=False))
    for k in ["input", "duration_s", "channels", "models", "config", "sections", "section_gains_db", "final_gain_db", "constants",
              "restore_amount", "residual_spans", "timings_s"]:
        assert k in rep
    assert rep["models"] == ["a", "b"]
    assert rep["channels"] == 1
    assert rep["duration_s"] == pytest.approx(12.0, abs=0.01)
    assert len(rep["sections"]) >= 1
    assert len(rep["section_gains_db"]) == len(rep["restore_amount"]) == len(rep["sections"])
    assert set(rep["timings_s"]) == {"load", "separate", "ensemble", "sections", "restore", "loudness",
                                     "residual", "write"}
    assert rep["config"]["strength"] == 0.5
    assert rep["final_gain_db"] <= 0.0
    assert {"sections", "restore", "loudness", "residual", "separate"} <= set(rep["constants"])
    assert rep["constants"]["sections"]["V_RATIO_THRESHOLD"] == 0.12
    assert rep["constants"]["residual"]["reduction"] == 0.8


def test_reusing_stems_skips_separation(tmp_path, song):
    path, seps = song()
    stems = separate_all(load_audio(path), seps)
    assert [s.calls for s in seps] == [1, 1]
    rep = run_pipeline(path, tmp_path / "o", seps, PipelineConfig(write_mp3=False), stems=stems)
    assert rep["timings_s"]["separate"] == 0.0
    assert rep["models"] == ["a", "b"]
    # 잔여 스캔이 첫 분리기를 한 번 더 부르는 것 외에 재분리는 없다.
    assert seps[1].calls == 1 and seps[0].calls == 2


def test_residual_disabled_skips_rescan(tmp_path, song):
    path, seps = song()
    stems = separate_all(load_audio(path), seps)
    rep = run_pipeline(path, tmp_path / "o", seps, PipelineConfig(write_mp3=False, residual=False), stems=stems)
    assert [s.calls for s in seps] == [1, 1]  # 잔여 스캔용 재분리가 없다
    assert rep["residual_spans"] == []
    assert rep["config"]["residual"] is False



def test_finalize_pair_uses_one_scale_for_both():
    """음량 보정으로 restored 피크만 커져도 두 파일을 같은 배율로 낮춰야 같은 시각 A/B 비교가 공정하다."""
    from mrengine.pipeline import finalize_pair

    base = np.full((2, 100), 0.5, dtype=np.float32)
    rest = base.copy()
    rest[:, 50:] = 2.0  # 보정된 구간
    b, r = finalize_pair(base, rest)
    assert b.shape == r.shape == (2, 100)
    assert np.abs(r).max() <= 10 ** (-1 / 20) + 1e-6
    np.testing.assert_allclose(r[:, :50], b[:, :50])  # 보정 밖은 그대로 같다
    np.testing.assert_allclose(r[:, 50:] / b[:, 50:], 4.0, rtol=1e-6)
