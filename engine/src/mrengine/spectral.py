"""STFT 공통 함수. 이후 구간 검출·복구·잔여 제거가 모두 이 프레임 배치를 공유한다."""
from __future__ import annotations

import numpy as np
from scipy.signal import istft as _istft
from scipy.signal import stft as _stft

from mrengine.audio_io import SR, fit_length

N_FFT = 4096
HOP = 1024


def stft(audio: np.ndarray) -> np.ndarray:
    """(channels, samples) -> complex64 (channels, freqs=N_FFT//2+1, frames).

    Hann 창, scipy 기본 경계(양끝 N_FFT//2 영 패딩). 크기는 scipy 스케일(spectrum)이라
    창 합으로 나뉘어 있다. istft가 이를 되돌린다.
    """
    _, _, z = _stft(audio, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, axis=-1)
    return z.astype(np.complex64)


def istft(spec: np.ndarray, n_samples: int) -> np.ndarray:
    """stft의 역변환. (channels, n_samples) float32, 길이는 n_samples로 맞춘다."""
    _, x = _istft(spec, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP)
    return fit_length(x.astype(np.float32), n_samples)


def frame_times(n_frames: int) -> np.ndarray:
    """각 프레임 중심의 시각(초). scipy 경계 패딩 때문에 프레임 k의 중심은 k*HOP 샘플이다."""
    return np.arange(n_frames) * HOP / SR
