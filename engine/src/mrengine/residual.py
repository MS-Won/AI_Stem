"""잔여 보컬 스캔과 감쇄.

완성된 MR을 다시 분리해 보컬 성분이 남은 구간을 찾고, 그 구간에서만 줄인다.
구간 밖은 입력과 비트 단위로 같다. 찾은 구간은 UI에 보여주기 위해 함께 반환한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from mrengine.audio_io import SR
from mrengine.separate import Separator

WINDOW_S = 0.5
FADE_S = 0.1


@dataclass(frozen=True)
class ResidualSpan:
    start_s: float
    end_s: float
    level_db: float

    def to_dict(self) -> dict:
        return asdict(self)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def scan_and_reduce(
    mr: np.ndarray,
    separator: Separator,
    threshold_db: float = -30.0,
    reduction: float = 0.8,
) -> tuple[np.ndarray, list[ResidualSpan]]:
    ref = _rms(mr)
    if ref == 0.0:
        return mr, []

    vocal = separator.separate(mr).vocals
    n = mr.shape[-1]
    win = int(round(WINDOW_S * SR))

    # 0.5초 창별 보컬 수준(MR 전체 RMS 대비 dB). 초과한 창을 이어 붙여 구간으로 만든다.
    spans: list[tuple[int, int, float]] = []
    for s in range(0, n, win):
        e = min(s + win, n)
        level = 20 * np.log10(max(_rms(vocal[..., s:e]), 1e-12) / ref)
        if level <= threshold_db:
            continue
        if spans and spans[-1][1] == s:
            spans[-1] = (spans[-1][0], e, max(spans[-1][2], level))
        else:
            spans.append((s, e, level))

    if not spans:
        return mr, []

    out = mr.copy()
    fade = int(round(FADE_S * SR))
    for s, e, _ in spans:
        i = np.arange(e - s)
        gain = np.minimum(1.0, np.minimum(i / fade, (e - s - i) / fade)) * reduction
        out[..., s:e] = mr[..., s:e] - (gain * vocal[..., s:e]).astype(np.float32)
    return out, [ResidualSpan(s / SR, e / SR, float(lv)) for s, e, lv in spans]
