"""분리 모델 교체 실험: 후보 모델별 반주를 끊김 수, 두께, 보컬 누출로 비교한다.

라이브 음원(소리도둑 유봉)에서 baseline이 뮤트처럼 끊기는 원인이 분리 모델이었다(STATE.md '1차 청취 판정').
같은 지표로 후보 모델을 비교해 DEFAULT_MODELS 교체 여부를 정한다.

    cd engine && .venv/Scripts/python scripts/model_trial.py --songs "../Test AR" --out out/phase0 \
        [--models a.ckpt,b.ckpt] [--only "14. 소리도둑 유봉 AR"]

- 분리 결과는 phase0_batch와 같은 out/phase0/<곡>/stems_cache/<모델>/ 에 저장해 재사용한다.
- 청취용 파일: out/phase0/<곡>/model_trial/<모델>.mp3 (피크가 -1dBFS를 넘을 때만 낮춤, 음량 보정 없음)
- 결과표: out/phase0/model_trial.md, model_trial.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mrengine.audio_io import load_audio, write_mp3  # noqa: E402
from mrengine.loudness import finalize  # noqa: E402
from mrengine.separate import DEFAULT_MODELS, MODEL_DIR, AudioSeparatorBackend  # noqa: E402
from phase0_batch import AUDIO_EXTS, _find_gaudio, _free_memory, load_or_separate  # noqa: E402

SR = 44100
CANDIDATES = [
    "melband_roformer_inst_v2.ckpt",
    "melband_roformer_instvox_duality_v2.ckpt",
    "MDX23C-8KFFT-InstVoc_HQ.ckpt",
]
LEAK_MODEL = "htdemucs_ft.yaml"  # 누출 측정용 고정 판정 모델 (htdemucs 자신은 유리하게 나올 수 있음)

DROP_WIN_S = 0.1       # 끊김 판정 창
DROP_CTX = 21          # 주변 중앙값 창 수 (약 2초)
DROP_DB = -12.0        # 주변보다 이만큼 꺼지면 끊김
MIX_STABLE_DB = -3.0   # 그때 원곡은 이 이상 꺼지지 않아야 함 (원곡 자체가 조용해진 곳 제외)
LOUD_RANGE_DB = 40.0   # 원곡 최대 대비 이 범위 안의 창만 '소리 있는 창'


def _rms_db(x: np.ndarray, w: int) -> np.ndarray:
    m = x.shape[1] // w
    x = x[:, : m * w].reshape(x.shape[0], m, w)
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2, axis=(0, 2)) + 1e-12)


def dropout_times(mr: np.ndarray, mix: np.ndarray) -> list[float]:
    """MR이 주변 2초보다 12dB 이상 갑자기 꺼지는데 원곡은 그렇지 않은 곳의 시작 시각들."""
    w = int(SR * DROP_WIN_S)
    n = min(mr.shape[1], mix.shape[1])
    dm, dr = _rms_db(mix[:, :n], w), _rms_db(mr[:, :n], w)
    loud = dm > dm.max() - LOUD_RANGE_DB
    m_loc = dm - median_filter(dm, size=DROP_CTX, mode="nearest")
    r_loc = dr - median_filter(dr, size=DROP_CTX, mode="nearest")
    hit = np.where(loud & (r_loc < DROP_DB) & (m_loc > MIX_STABLE_DB))[0]
    times: list[float] = []
    last = -10
    for i in hit:
        if i - last > 1:
            times.append(float(i * DROP_WIN_S))
        last = i
    return times


def thickness_db(mr: np.ndarray, mix: np.ndarray) -> float:
    """소리 있는 0.5초 창에서 MR-원곡 dB 중앙값. 클수록 반주가 두껍다(보컬이 남아도 커진다)."""
    w = SR // 2
    n = min(mr.shape[1], mix.shape[1])
    dm, dr = _rms_db(mix[:, :n], w), _rms_db(mr[:, :n], w)
    loud = dm > dm.max() - LOUD_RANGE_DB
    return float(np.median((dr - dm)[loud]))


def leak_db(mr: np.ndarray, judge: AudioSeparatorBackend) -> float:
    """MR을 판정 모델로 다시 분리했을 때 보컬 에너지/MR 에너지 (dB). 작을수록 보컬이 덜 남았다."""
    v = judge.separate(mr).vocals
    ev = float(np.sum(np.square(v, dtype=np.float64)))
    em = float(np.sum(np.square(mr, dtype=np.float64)))
    return 10 * np.log10((ev + 1e-12) / (em + 1e-12))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--songs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS + CANDIDATES))
    p.add_argument("--only", default=None, help="이 이름으로 시작하는 곡만")
    args = p.parse_args()
    models = [m for m in args.models.split(",") if m]

    songs = sorted(x for x in args.songs.iterdir() if x.is_file() and x.suffix.lower() in AUDIO_EXTS)
    if args.only:
        songs = [s for s in songs if s.stem.startswith(args.only)]
    res_path = args.out / "model_trial.json"
    results: dict = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}

    for song in songs:
        print(f"== {song.name}", flush=True)
        mix = load_audio(song)
        song_dir = args.out / song.stem
        trial_dir = song_dir / "model_trial"
        trial_dir.mkdir(parents=True, exist_ok=True)
        judge = None
        rows = results.setdefault(song.stem, {})
        cands: list[tuple[str, np.ndarray, float | None]] = []
        for m in models:
            t0 = time.perf_counter()
            stems, spent = load_or_separate(mix, song_dir, [m])
            cands.append((m, stems[0].instrumental, spent if spent > 0 else None))
            print(f"   {m}: 분리 {spent:.0f}초 ({time.perf_counter() - t0:.0f}초)", flush=True)
            _free_memory()
        g = _find_gaudio(args.songs, song.stem)
        if g is not None:
            cands.append(("Gaudio", load_audio(g), None))
        for name, mr, spent in cands:
            if judge is None:
                judge = AudioSeparatorBackend(LEAK_MODEL, MODEL_DIR, song_dir / "_work")
            n = min(mr.shape[1], mix.shape[1])
            mr = mr[:, :n] if mr.shape[0] == mix.shape[0] else np.broadcast_to(mr[:1, :n], (mix.shape[0], n)).copy()
            drops = dropout_times(mr, mix)
            row = {"dropouts": len(drops), "dropout_examples": drops[:15],
                   "thickness_db": thickness_db(mr, mix), "leak_db": leak_db(mr, judge)}
            if spent is not None:
                row["separate_s"] = spent
            elif name in rows and "separate_s" in rows[name]:
                row["separate_s"] = rows[name]["separate_s"]
            rows[name] = row
            if name != "Gaudio":
                write_mp3(trial_dir / f"{Path(name).stem}.mp3", finalize(mr))
            print(f"   {name:45s} 끊김 {row['dropouts']:3d}  두께 {row['thickness_db']:6.1f}dB  "
                  f"누출 {row['leak_db']:6.1f}dB", flush=True)
            _free_memory()
        del judge
        _free_memory()
        res_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 분리 모델 교체 실험", "",
             "- 끊김: 0.1초 창에서 MR이 주변 2초 중앙값보다 12dB 이상 꺼지고 원곡은 3dB 이내인 곳의 수 (적을수록 좋음)",
             "- 두께: 소리 있는 0.5초 창의 MR-원곡 dB 중앙값 (클수록 반주가 두꺼움, 보컬이 남아도 커짐)",
             f"- 누출: MR을 `{LEAK_MODEL}`로 다시 분리한 보컬 에너지/MR 에너지 (작을수록 좋음, htdemucs 자신은 유리할 수 있음)",
             "- 청취 파일: `out/phase0/<곡>/model_trial/<모델>.mp3`", ""]
    for song, rows in results.items():
        lines += [f"## {song}", "", "| 모델 | 끊김 | 두께 | 누출 | 분리 시간 | 끊김 예 |", "|---|---|---|---|---|---|"]
        for name, r in rows.items():
            ex = ", ".join(f"{int(t // 60):02d}:{t % 60:04.1f}" for t in r["dropout_examples"][:6])
            sp = f"{r['separate_s']:.0f}초" if "separate_s" in r else "-"
            lines.append(f"| {name} | {r['dropouts']} | {r['thickness_db']:.1f}dB | {r['leak_db']:.1f}dB | {sp} | {ex} |")
        lines.append("")
    (args.out / "model_trial.md").write_text("\n".join(lines), encoding="utf-8")
    print("결과표:", (args.out / "model_trial.md").resolve())


if __name__ == "__main__":
    main()
