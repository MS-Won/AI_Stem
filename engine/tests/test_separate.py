import numpy as np
import pytest
from conftest import rms

from mrengine.separate import (
    DEFAULT_MODELS,
    MODEL_DIR,
    AudioSeparatorBackend,
    Stems,
    normalize_stems,
)

SR = 44100


def test_normalize_stems_restores_mono_from_stereo(tone):
    mono_mix = tone(440, 1.0, 0.5)
    left = tone(440, 1.0, 0.4)[0]
    right = tone(440, 1.0, 0.2)[0]
    stereo = np.stack([left, right])
    stems = normalize_stems(stereo, stereo * 0.5, like=mono_mix)
    assert stems.vocals.shape == mono_mix.shape
    assert stems.instrumental.shape == mono_mix.shape
    np.testing.assert_allclose(stems.vocals[0], (left + right) / 2, atol=1e-6)


def test_normalize_stems_duplicates_mono_to_stereo(tone):
    stereo_mix = np.vstack([tone(440, 1.0), tone(440, 1.0)])
    mono_out = tone(300, 1.0, 0.3)
    stems = normalize_stems(mono_out, mono_out, like=stereo_mix)
    assert stems.vocals.shape == stereo_mix.shape
    np.testing.assert_array_equal(stems.vocals[0], stems.vocals[1])


@pytest.mark.parametrize("model_len", [SR - 500, SR + 700])
def test_normalize_stems_trims_or_pads_to_input_length(tone, model_len):
    mix = tone(440, 1.0)
    raw = np.full((1, model_len), 0.1, dtype=np.float32)
    stems = normalize_stems(raw, raw, like=mix)
    assert stems.vocals.shape == mix.shape == stems.instrumental.shape
    assert stems.vocals.dtype == np.float32
    if model_len < SR:
        assert stems.vocals[0, -1] == 0.0  # 짧으면 0으로 채움


def test_default_models_are_two_known_files():
    assert len(DEFAULT_MODELS) == 2
    assert MODEL_DIR.name == "models" and MODEL_DIR.parent.name == "engine"


@pytest.mark.gpu
@pytest.mark.parametrize("model", DEFAULT_MODELS)
def test_separate_handles_unicode_and_hash_path(tmp_path, tone, model):
    work = tmp_path / "작업 #1"
    be = AudioSeparatorBackend(model, MODEL_DIR, work)
    assert be.name == model
    mix = tone(440, 5.0, 0.5)
    stems = be.separate(mix)
    assert isinstance(stems, Stems)
    assert stems.vocals.shape == mix.shape
    assert stems.instrumental.shape == (1, 5 * SR)
    assert stems.instrumental.dtype == np.float32


@pytest.mark.gpu
def test_real_model_separates_noise_from_tone(tmp_path, tone, band_noise):
    # 합성 신호에서 결과가 나오는지만 확인한다. 품질 판정은 Task 9 청취.
    mix = tone(220, 5.0, 0.3) + band_noise(2000, 6000, 5.0, 0.3)
    be = AudioSeparatorBackend(DEFAULT_MODELS[0], MODEL_DIR, tmp_path / "w")
    stems = be.separate(mix)
    assert rms(stems.vocals) + rms(stems.instrumental) > 0.01
    assert np.isfinite(stems.vocals).all() and np.isfinite(stems.instrumental).all()
