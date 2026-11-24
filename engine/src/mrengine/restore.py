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
FRAME_CHUNK = 512          # 복구 계산을 한 번에 처리할 프레임 수
REF_BIN_CHUNK = 256       # ref median 계산 시 한 번에 처리할 주파수 bin 수


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


def _reference_ratio(M: np.ndarray, I: np.ndarray, quiet: np.ndarray) -> np.ndarray:
    """주파수별 '정상 반주 비율' median(|I| / |M|), quiet 프레임 기준. M, I: (1, freqs, frames) -> (freqs,)

    메모리를 아끼려고 주파수 대역을 나눠 처리한다."""
    n_bins = M.shape[1]
    ref = np.empty(n_bins, dtype=np.float32)
    for lo in range(0, n_bins, REF_BIN_CHUNK):
        hi = min(lo + REF_BIN_CHUNK, n_bins)
        ratio = np.abs(I[:, lo:hi, quiet]) / (np.abs(M[:, lo:hi, quiet]) + EPS)
        ref[lo:hi] = np.median(ratio, axis=(0, 2))
    return ref


def restore_choir_bins(M: np.ndarray, I: np.ndarray, V: np.ndarray, ref: np.ndarray, s: np.ndarray) -> np.ndarray:
    """칸별 복구 (touched 프레임만 잘라 넣는다). M, I, V: complex (ch, freqs, frames), ref: (freqs,),
    s: (frames,) 프레임별 strength. -> I' (|I'| <= |M| 이 모든 칸에서 성립)."""
    R = M - V
    M_mag, I_mag = np.abs(M), np.abs(I)
    target = np.minimum(ref[None, :, None] * M_mag, np.abs(R))
    new_mag = I_mag + s[None, None, :] * np.maximum(target - I_mag, 0.0)
    use_r = I_mag < PHASE_SWITCH * target
    phase = np.exp(1j * np.where(use_r, np.angle(R), np.angle(I))).astype(np.complex64)
    new = new_mag * phase
    new *= np.minimum(1.0, M_mag / (np.abs(new) + EPS)).astype(np.float32)
    return new


def restore_choir(
    mix: np.ndarray,
    inst: np.ndarray,
    vocals: np.ndarray,
    sections: list[Section],
    strength: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """choir 구간의 inst 빈 곳을 믹스-보컬 나머지로 채운다. -> (복구된 inst, 프레임별 복구량 비율)."""
    out = inst.copy()
    amount_parts = None
    for c in range(inst.shape[0]):
        # 채널별로 처리해 곡 길이가 길어도 STFT 3개를 동시에 들고 있는 양을 절반으로 줄인다.
        M, I = stft(mix[c : c + 1]), stft(inst[c : c + 1])
        n_frames = M.shape[-1]
        if amount_parts is None:
            s = _strength_curve(sections, frame_times(n_frames), strength)
            idx = np.flatnonzero(s > 0)
            if idx.size == 0:
                return out, np.zeros(n_frames, dtype=np.float32)
            quiet = np.flatnonzero(~frames_mask(sections, n_frames, {"choir", "lead"}))
            if quiet.size < MIN_REF_FRAME_FRACTION * n_frames:
                quiet = np.arange(n_frames)
            touched = np.zeros(n_frames, dtype=bool)
            touched[idx] = True
            amount_parts = np.zeros((3, n_frames), dtype=np.float64)  # e_new, e_old, e_mix
        ref = _reference_ratio(M, I, quiet)
        V = stft(vocals[c : c + 1])

        # 프레임을 덩어리로 나눠 임시 배열이 곡 길이에 비례해 커지지 않게 한다.
        # I는 제자리에서 '복구 차이(I' - I)'로 바꿔 쓴다 (바뀌지 않은 프레임은 0).
        for lo in range(0, idx.size, FRAME_CHUNK):
            ci = idx[lo : lo + FRAME_CHUNK]
            Ms, Is = M[:, :, ci], I[:, :, ci]
            new = restore_choir_bins(Ms, Is, V[:, :, ci], ref, s[ci])
            amount_parts[0, ci] += np.sum(np.abs(new) ** 2, axis=(0, 1), dtype=np.float64)
            amount_parts[1, ci] += np.sum(np.abs(Is) ** 2, axis=(0, 1), dtype=np.float64)
            amount_parts[2, ci] += np.sum(np.abs(Ms) ** 2, axis=(0, 1), dtype=np.float64)
            I[:, :, ci] = new - Is
        del M, V
        I[:, :, ~touched] = 0
        # 바뀐 프레임만 담은 차이를 더해, 구간 밖은 inst 그대로 보존한다 (STFT 왕복 오차 방지).
        out[c] = inst[c] + istft(I, inst.shape[-1])[0]
        del I

    e_new, e_old, e_mix = amount_parts
    amount = np.maximum((e_new - e_old) / (e_mix + EPS), 0.0).astype(np.float32)
    return out.astype(np.float32), amount
