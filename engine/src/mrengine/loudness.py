"""구간 음량 복구와 최종 피크 안전 처리.

합창 구간에서는 보컬을 지우면서 반주도 같이 작아지는 경우가 많아, 사용자가 손으로 키우던 작업을
자동화한다. 키우는 양은 max_gain_db로 제한해 남은 보컬 잔여분이 같이 커지지 않게 한다.
"""
from __future__ import annotations

import numpy as np

from mrengine.audio_io import SR
from mrengine.sections import Section

REF_WINDOW_S = 3.0
MIN_REF_S = 0.5
RAMP_S = 0.2


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def _span(section: Section, n: int) -> tuple[int, int]:
    return max(0, int(round(section.start_s * SR))), min(n, int(round(section.end_s * SR)))


def _section_gain_db(inst, mix, idx, sections, max_gain_db) -> float:
    n = inst.shape[-1]
    s, e = _span(sections[idx], n)
    if e <= s:
        return 0.0
    in_mix = _rms(mix[:, s:e])
    in_inst = _rms(inst[:, s:e])
    if in_mix < 1e-9 or in_inst < 1e-9:
        return 0.0
    w = int(round(REF_WINDOW_S * SR))
    ref = np.zeros(n, dtype=bool)
    ref[max(0, s - w) : s] = True
    ref[e : min(n, e + w)] = True
    for j, other in enumerate(sections):  # 다른 보컬 구간(choir/lead) 제외
        if j != idx:
            os_, oe = _span(other, n)
            ref[os_:oe] = False
    if ref.sum() < MIN_REF_S * SR:
        return 0.0
    ref_mix = _rms(mix[:, ref])
    if ref_mix < 1e-9:
        return 0.0
    ref_ratio = _rms(inst[:, ref]) / ref_mix
    gain = 20 * np.log10(max(ref_ratio, 1e-9) / (in_inst / in_mix))
    return float(np.clip(gain, 0.0, max_gain_db))


def restore_section_loudness(
    inst: np.ndarray, mix: np.ndarray, sections: list[Section], max_gain_db: float = 6.0
) -> tuple[np.ndarray, list[float]]:
    """choir 구간의 inst/mix 음량비를 주변 기준 비율까지 올린다 (내리지 않음, 상한 max_gain_db)."""
    n = inst.shape[-1]
    gains = [
        _section_gain_db(inst, mix, i, sections, max_gain_db) if sec.kind == "choir" else 0.0
        for i, sec in enumerate(sections)
    ]
    env_db = np.zeros(n, dtype=np.float64)
    ramp = int(round(RAMP_S * SR))
    for sec, g in zip(sections, gains):
        s, e = _span(sec, n)
        if g <= 0.0 or e <= s:
            continue
        pos = np.arange(e - s)
        up = np.minimum(pos / ramp, 1.0)
        down = np.minimum((e - s - pos) / ramp, 1.0)
        seg = g * np.minimum(up, down)
        env_db[s:e] = np.maximum(env_db[s:e], seg)
    out = (inst * (10.0 ** (env_db / 20.0))).astype(np.float32)
    return out, gains


def finalize(audio: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """피크가 ceiling을 넘으면 전체를 같은 비율로 낮춘다 (리미터 아님). 넘지 않으면 그대로."""
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    limit = 10.0 ** (ceiling_db / 20.0)
    if peak <= limit:
        return audio
    out = (audio.astype(np.float64) * (limit / peak)).astype(np.float32)
    return out
