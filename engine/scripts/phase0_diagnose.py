"""합창 구간 복구량이 왜 작은지 진단한다 (phase0_batch.py가 만든 분리 캐시 사용).

    python scripts/phase0_diagnose.py "out/phase0/<곡>" "<원곡 경로>" [--vocals avg|min_mag|<모델>]

choir 구간마다 (채널 0 기준):
  hole_db   : 구간 안 inst 에너지 / (ref*|M|)^2 에너지 (dB). 0에 가까우면 '정상 반주 비율'만큼 이미 있다.
  i_vs_r_db : inst 에너지 / |M-V|^2 에너지 (dB). 0이면 inst가 이미 '믹스-보컬'과 같다 -> 채울 재료가 없다.
  r_bind    : target = min(ref|M|, |R|)에서 |R|이 더 작은(묶는) 칸 비율.
  deficit   : sum(max(target-|I|,0)^2) / sum(|I|^2) — strength=1에서 더할 수 있는 크기 차이의 에너지 비.
  gain1_db  : strength=1로 restore_choir_bins를 돌렸을 때 구간 inst 에너지 증가 (dB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mrengine.audio_io import load_audio
from mrengine.ensemble import ensemble
from mrengine.restore import _reference_ratio, restore_choir_bins
from mrengine.sections import detect_vocal_sections, frames_mask
from mrengine.separate import DEFAULT_MODELS
from mrengine.spectral import frame_times, stft

EPS = 1e-12


def _db(a: float, b: float) -> float:
    return float(10 * np.log10((a + EPS) / (b + EPS)))


def diagnose(song_dir: Path, song: Path, vocals_mode: str, models: list[str]) -> dict:
    mix = load_audio(song)
    n = mix.shape[1]
    cache = [(np.load(song_dir / "stems_cache" / m / "vocals.npy"),
              np.load(song_dir / "stems_cache" / m / "instrumental.npy")) for m in models]
    inst = ensemble([c[1] for c in cache], n, "avg")
    v_avg = ensemble([c[0] for c in cache], n, "avg")
    sections = detect_vocal_sections(v_avg, mix)
    if vocals_mode in ("avg", "min_mag", "max_mag"):
        vocals = ensemble([c[0] for c in cache], n, vocals_mode)
    else:
        vocals = cache[models.index(vocals_mode)][0]

    # 모델 출력이 '믹스 - 보컬'로 만들어졌는지(시간 영역) 확인
    sub = {m: _db(float(np.sum((mix - v - i) ** 2, dtype=np.float64)), float(np.sum(i ** 2, dtype=np.float64)))
           for m, (v, i) in zip(models, cache)}

    M, I, V = stft(mix[:1]), stft(inst[:1]), stft(vocals[:1])
    nf = M.shape[-1]
    quiet = np.flatnonzero(~frames_mask(sections, nf, {"choir", "lead"}))
    ref = _reference_ratio(M, I, quiet)
    t = frame_times(nf)
    rows = []
    for sec in sections:
        if sec.kind != "choir":
            continue
        sel = np.flatnonzero((t >= sec.start_s) & (t < sec.end_s))
        Ms, Is, Vs = M[:, :, sel], I[:, :, sel], V[:, :, sel]
        Mm, Im, Rm = np.abs(Ms), np.abs(Is), np.abs(Ms - Vs)
        refM = ref[None, :, None] * Mm
        target = np.minimum(refM, Rm)
        e_i = float(np.sum(Im ** 2, dtype=np.float64))
        new = restore_choir_bins(Ms, Is, Vs, ref, np.ones(len(sel), dtype=np.float32))
        rows.append({
            "start_s": round(sec.start_s, 2), "dur_s": round(sec.end_s - sec.start_s, 2),
            "hole_db": round(_db(e_i, float(np.sum(refM ** 2, dtype=np.float64))), 2),
            "i_vs_r_db": round(_db(e_i, float(np.sum(Rm ** 2, dtype=np.float64))), 2),
            "i_vs_m_db": round(_db(e_i, float(np.sum(Mm ** 2, dtype=np.float64))), 2),
            "r_bind": round(float(np.mean(Rm < refM)), 3),
            "deficit": round(float(np.sum(np.maximum(target - Im, 0) ** 2, dtype=np.float64)) / (e_i + EPS), 4),
            "gain1_db": round(_db(float(np.sum(np.abs(new) ** 2, dtype=np.float64)), e_i), 2),
        })
    keys = ["hole_db", "i_vs_r_db", "i_vs_m_db", "r_bind", "deficit", "gain1_db"]
    w = np.array([r["dur_s"] for r in rows]) if rows else np.zeros(0)
    summary = {k: round(float(np.average([r[k] for r in rows], weights=w)), 3) for k in keys} if rows else {}
    return {
        "vocals_mode": vocals_mode,
        "n_sections": len(sections), "n_choir": len(rows),
        "quiet_frame_fraction": round(quiet.size / nf, 3),
        "ref_median_200_8k": round(float(np.median(ref[(np.fft.rfftfreq(4096, 1 / 44100) >= 200)
                                                        & (np.fft.rfftfreq(4096, 1 / 44100) <= 8000)])), 3),
        "mix_minus_v_minus_i_db": {m: round(v, 1) for m, v in sub.items()},
        "weighted_mean": summary,
        "sections": rows,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("song_dir", type=Path)
    p.add_argument("song", type=Path)
    p.add_argument("--vocals", default="avg")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    a = p.parse_args()
    models = [m for m in a.models.split(",") if m]
    out = diagnose(a.song_dir, a.song, a.vocals, models)
    print(json.dumps({k: v for k, v in out.items() if k != "sections"}, ensure_ascii=False, indent=1))
    (a.song_dir / f"diagnose_{a.vocals}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
