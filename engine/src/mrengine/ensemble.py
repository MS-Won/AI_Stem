"""여러 분리 모델의 MR을 STFT 영역에서 합친다."""
from __future__ import annotations

from typing import Literal

import numpy as np

from mrengine.audio_io import fit_length
from mrengine.spectral import istft, stft

EnsembleMethod = Literal["avg", "min_mag", "max_mag"]


def ensemble(mrs: list[np.ndarray], n_samples: int, method: EnsembleMethod = "avg") -> np.ndarray:
    """avg는 복소 평균, min_mag/max_mag는 칸마다 크기가 가장 작은/큰 모델의 복소값을 그대로 쓴다."""
    if not mrs:
        raise ValueError("mrs is empty")
    if method not in ("avg", "min_mag", "max_mag"):
        raise ValueError(f"unknown method: {method}")
    # 하나씩 누적해 complex64 한 장 + 후보 한 장만 메모리에 둔다
    acc = None
    for mr in mrs:
        s = stft(fit_length(mr, n_samples))
        if acc is None:
            acc = s
        elif method == "avg":
            acc += s
        else:
            pick = np.abs(s) < np.abs(acc) if method == "min_mag" else np.abs(s) > np.abs(acc)
            acc = np.where(pick, s, acc)
    if method == "avg":
        acc /= len(mrs)
    return istft(acc, n_samples)
