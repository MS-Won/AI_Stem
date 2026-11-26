"""오디오 입출력. 배열 형식은 항상 float32 (channels, samples), 44100Hz."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR: int = 44100
_SF_EXTS = {".wav", ".flac"}


def _channels(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels", "-of", "csv=p=0", str(path)],
        capture_output=True, check=True,
    ).stdout.decode().strip()
    return int(out)


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g, axis=1).astype(np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    """(channels, samples) float32, 44100Hz. 채널 수는 원본을 보존한다."""
    path = Path(path)
    if path.suffix.lower() in _SF_EXTS:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        x = np.ascontiguousarray(data.T)
        return x if sr == SR else _resample(x, sr)
    ch = _channels(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-ac", str(ch), "-ar", str(SR), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype="<f4").reshape(-1, ch).T.copy()


def fit_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """n_samples에 맞게 자르거나 끝에 0을 채운다."""
    n = audio.shape[1]
    if n >= n_samples:
        return audio[:, :n_samples]
    return np.pad(audio, ((0, 0), (0, n_samples - n)))


def write_wav(path: str | Path, audio: np.ndarray) -> None:
    """16-bit PCM WAV."""
    sf.write(str(path), audio.T, SR, subtype="PCM_16")


def write_mp3(path: str | Path, audio: np.ndarray) -> None:
    """ffmpeg로 320kbps MP3 (임시 WAV 경유)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "in.wav"
        write_wav(tmp, audio)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-b:a", "320k", str(path)],
            capture_output=True, check=True,
        )
