"""합창 구간 반주 복구.

분리 모델은 합창 구간에서 악기 에너지를 같이 지워 MR이 얇아진다(spectral hole).
여기서는 원곡 믹스에서 보컬 추정치를 뺀 나머지(R)만을 재료로 빈 곳을 다시 채운다.
어떤 칸에서도 믹스 크기를 넘지 않으므로 믹스에 없던 소리를 만들어내지 않는다.
"""
from __future__ import annotations

import numpy as np

from mrengine.sections import Section, frames_mask
from mrengine.spectral import frame_times, istft, stft

EDGE_RAMP_S = 0.1          # 구간 경계에서 strength를 0에서 선형으로 올리는 길이
MIN_REF_FRAME_FRACTION = 0.1  # 보컬 없는 프레임이 이보다 적으면 곡 전체 median으로 대체
PHASE_SWITCH = 0.25        # |I|가 target의 이 비율 미만이면 R의 위상을 쓴다
EPS = 1e-8
CLAMP_PASSES = 20           # 재클램프 최대 횟수 (합성 테스트에서 10회로는 1e-3 초과가 남고 20회에서 수렴)
CLAMP_TOL = 1e-4           # 이 정도 초과는 무시 (float32 오차)


def _strength_curve(sections: list[Section], times: np.ndarray, strength: float) -> np.ndarray:
    """프레임별 유효 strength. choir 구간 안쪽 0.1초에서만 0 -> strength로 오르내린다."""
    s = np.zeros(len(times), dtype=np.float32)
    for sec in sections:
        if sec.kind != "choir":
            continue
        inside = (times >= sec.start_s) & (times < sec.end_s)
        up = np.clip((times - sec.start_s) / EDGE_RAMP_S, 0.0, 1.0)
        down = np.clip((sec.end_s - times) / EDGE_RAMP_S, 0.0, 1.0)
        s = np.maximum(s, np.where(inside, strength * np.minimum(up, down), 0.0).astype(np.float32))
    return s


def _reference_ratio(M_mag: np.ndarray, I_mag: np.ndarray, sections: list[Section]) -> np.ndarray:
    """주파수별 '정상 반주 비율' median(|I| / |M|), 보컬 없는 프레임 기준. -> (freqs,)"""
    n_frames = M_mag.shape[-1]
    quiet = ~frames_mask(sections, n_frames, {"choir", "lead"})
    if quiet.sum() < MIN_REF_FRAME_FRACTION * n_frames:
        quiet = np.ones(n_frames, dtype=bool)
    ratio = I_mag[:, :, quiet] / (M_mag[:, :, quiet] + EPS)
    return np.median(ratio, axis=(0, 2)).astype(np.float32)


def restore_choir(
    mix: np.ndarray,
    inst: np.ndarray,
    vocals: np.ndarray,
    sections: list[Section],
    strength: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """choir 구간의 inst 빈 곳을 믹스-보컬 나머지로 채운다. -> (복구된 inst, 프레임별 복구량 비율)."""
    M, I, V = stft(mix), stft(inst), stft(vocals)
    n_frames = M.shape[-1]
    s = _strength_curve(sections, frame_times(n_frames), strength)
    amount = np.zeros(n_frames, dtype=np.float32)
    idx = np.flatnonzero(s > 0)
    if idx.size == 0:
        return inst.copy(), amount

    ref = _reference_ratio(np.abs(M), np.abs(I), sections)[None, :, None]

    Ms, Is = M[:, :, idx], I[:, :, idx]
    Rs = Ms - V[:, :, idx]
    del V
    M_mag, I_mag, R_mag = np.abs(Ms), np.abs(Is), np.abs(Rs)

    target = np.minimum(ref * M_mag, R_mag)
    deficit = np.maximum(target - I_mag, 0.0)
    new_mag = I_mag + s[idx][None, None, :] * deficit

    use_r = I_mag < PHASE_SWITCH * target
    phase = np.where(use_r, np.exp(1j * np.angle(Rs)), np.exp(1j * np.angle(Is))).astype(np.complex64)
    new = new_mag * phase

    # 칸마다 |I'| <= |M|
    new *= np.minimum(1.0, M_mag / (np.abs(new) + EPS)).astype(np.float32)

    e_new = np.sum(np.abs(new) ** 2, axis=(0, 1), dtype=np.float64)
    e_old = np.sum(I_mag**2, axis=(0, 1), dtype=np.float64)
    e_mix = np.sum(M_mag**2, axis=(0, 1), dtype=np.float64)
    amount[idx] = np.maximum((e_new - e_old) / (e_mix + EPS), 0.0)

    # 바뀐 프레임만 담은 차이를 더해, 구간 밖은 inst 그대로 보존한다 (STFT 왕복 오차 방지).
    delta = np.zeros_like(I)
    delta[:, :, idx] = new - Is
    out = inst + istft(delta, inst.shape[-1])
    del delta

    # istft->stft는 일관성 투영이라 클램프한 크기가 조금 되살아난다. 바뀐 프레임에서만 몇 번 다시 눌러 준다.
    for _ in range(CLAMP_PASSES):
        O = stft(out)
        Os = O[:, :, idx]
        over = np.abs(Os) > M_mag + CLAMP_TOL
        if not over.any():
            break
        fixed = Os * np.minimum(1.0, M_mag / (np.abs(Os) + EPS)).astype(np.float32)
        d = np.zeros_like(O)
        d[:, :, idx] = fixed - Os
        out = out + istft(d, inst.shape[-1])
    return out.astype(np.float32), amount
