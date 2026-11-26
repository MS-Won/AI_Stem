import numpy as np
import soundfile as sf

from mrengine.audio_io import fit_length, load_audio, write_mp3, write_wav


def test_load_mono_keeps_one_channel(tmp_path, tone):
    p = tmp_path / "mono.wav"
    write_wav(p, tone(440, 1.0, 0.5))
    a = load_audio(p)
    assert a.shape[0] == 1 and a.dtype == np.float32


def test_load_resamples_to_44100(tmp_path):
    p = tmp_path / "st22k.wav"
    t = np.arange(22050) / 22050
    x = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)], axis=1) * 0.5
    sf.write(p, x, 22050)
    a = load_audio(p)
    assert a.shape == (2, 44100)


def test_load_mp3_with_unicode_hash_path(tmp_path, tone):
    p = tmp_path / "#13 낡은 침대.mp3"
    write_mp3(p, tone(440, 1.0, 0.5))
    a = load_audio(p)
    assert a.shape[0] == 1 and abs(a.shape[1] - 44100) < 2048  # MP3 인코더 패딩 허용


def test_fit_length_pads_and_trims(tone):
    x = tone(440, 1.0, 0.5)
    assert fit_length(x, 100).shape == (1, 100)
    y = fit_length(x, 50000)
    assert y.shape == (1, 50000) and np.all(y[:, 44100:] == 0)


def test_write_wav_is_16bit(tmp_path, tone):
    p = tmp_path / "o.wav"
    write_wav(p, tone(440, 0.1, 0.5))
    assert sf.info(p).subtype == "PCM_16"
