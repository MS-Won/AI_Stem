"""보컬 구간 검출과 합창(choir)/리드(lead) 분류.

Task 5(합창 복구)와 Task 6(음량)은 choir 구간에서만 동작하므로, 여기서 choir를 놓치면
핵심 기능이 조용히 아무것도 안 한다. 임계값은 모두 모듈 상수이며 Task 9에서 실제 곡으로 조정한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import median_filter

from mrengine.audio_io import SR
from mrengine.spectral import HOP, N_FFT, frame_times, stft

V_RATIO_THRESHOLD = 0.25      # 프레임의 보컬/믹스 에너지 비가 이 값 초과면 보컬 활성
CHOIR_ABS_WIDTH_HZ = 3000.0   # 대역 폭이 이 값 이상이면 곡과 무관하게 choir (합창 위주 곡 대비)
CHOIR_WIDTH_RATIO = 1.5       # 대역 폭이 곡 기준 폭의 이 배수 이상이면 choir
REF_WIDTH_PERCENTILE = 20     # 곡 기준 폭 = 활성 프레임 폭의 이 백분위 (중앙값은 합창 위주 곡에서 올라가 버린다)
CHOIR_FLATNESS_THRESHOLD = 0.3  # 스펙트럼 평탄도가 이 값 이상이면 choir
BAND_LO_HZ = 200.0
BAND_HI_HZ = 8000.0
ENERGY_FRACTION = 0.8         # 대역 폭 = 보컬 에너지의 이 비율이 들어가는 최소 주파수 폭
MEDIAN_FILTER_S = 0.5

Kind = Literal["choir", "lead"]


@dataclass(frozen=True)
class Section:
    start_s: float
    end_s: float
    kind: Kind

    def to_dict(self) -> dict:
        return asdict(self)


def _bandwidth_and_flatness(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """power: (freqs, frames) 보컬 파워(채널 합), 200Hz~8kHz 대역만. -> (폭 Hz, 평탄도) 프레임별."""
    n_bins, n_frames = power.shape
    df = SR / N_FFT
    cum = np.cumsum(power, axis=0)
    total = cum[-1]
    flatness = np.exp(np.mean(np.log(power + 1e-20), axis=0)) / (np.mean(power, axis=0) + 1e-20)
    widths = np.zeros(n_frames)
    for k in range(n_frames):
        if total[k] <= 0:
            continue
        c = np.concatenate([[0.0], cum[:, k]])
        # 각 시작 bin i에서 80%에 도달하는 최소 끝 bin -> 최소 폭
        end = np.searchsorted(c, c[:-1] + ENERGY_FRACTION * total[k], side="left")
        ok = end <= n_bins
        widths[k] = (end[ok] - np.arange(n_bins)[ok]).min() * df
    return widths, flatness


def detect_vocal_sections(vocals: np.ndarray, mix: np.ndarray, min_len_s: float = 1.0) -> list[Section]:
    V = stft(vocals)
    M = stft(mix)
    n_frames = V.shape[-1]
    pv = (np.abs(V) ** 2).astype(np.float64)
    pm = (np.abs(M) ** 2).astype(np.float64)
    v_ratio = pv.sum((0, 1)) / (pm.sum((0, 1)) + 1e-20)

    win = max(1, int(round(MEDIAN_FILTER_S * SR / HOP)) | 1)
    active = median_filter((v_ratio > V_RATIO_THRESHOLD).astype(np.uint8), size=win, mode="nearest").astype(bool)
    if not active.any():
        return []

    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    band = (freqs >= BAND_LO_HZ) & (freqs <= BAND_HI_HZ)
    width, flat = _bandwidth_and_flatness(pv[:, band, :].sum(0))
    ref_width = np.percentile(width[active], REF_WIDTH_PERCENTILE)
    wide = width >= CHOIR_ABS_WIDTH_HZ
    if ref_width > 0:
        wide |= width >= CHOIR_WIDTH_RATIO * ref_width
    # 비활성 프레임의 폭·평탄도는 의미 없으므로 평활 전에 제외한다
    choir = active & (wide | (flat >= CHOIR_FLATNESS_THRESHOLD))
    choir = median_filter(choir.astype(np.uint8), size=win, mode="nearest").astype(bool)

    label = np.zeros(n_frames, dtype=np.int8)  # 0 없음, 1 lead, 2 choir
    label[active] = np.where(choir[active], 2, 1)

    t = frame_times(n_frames + 1)  # 마지막 프레임의 끝 경계용
    sections: list[Section] = []
    start = 0
    for k in range(1, n_frames + 1):
        if k == n_frames or label[k] != label[start]:
            if label[start] != 0 and t[k] - t[start] >= min_len_s:
                sections.append(Section(float(t[start]), float(t[k]), "choir" if label[start] == 2 else "lead"))
            start = k
    return sections


def frames_mask(sections: list[Section], n_frames: int, kinds: set[str]) -> np.ndarray:
    t = frame_times(n_frames)
    mask = np.zeros(n_frames, dtype=bool)
    for s in sections:
        if s.kind in kinds:
            mask |= (t >= s.start_s) & (t < s.end_s)
    return mask
