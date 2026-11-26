"""Phase 0 일괄 처리: 곡마다 분리를 한 번만 하고 strength x ensemble 조합을 돌려 비교 자료를 만든다.

    python scripts/phase0_batch.py --songs "../Test AR" --out out/phase0 \
        [--strengths 0.3,0.5,0.8,1.0] [--ensembles avg,min_mag] [--final 1.0,avg] [--cache-only] [--reuse]

- 분리 결과는 out/phase0/<곡>/stems_cache/<모델>/{vocals,instrumental}.npy 에 저장하고, 있으면 읽는다.
- 탐색 조합은 잔여 보컬 스캔(재분리, 가장 느림)을 건너뛰고(PipelineConfig.residual=False),
  --final 조합 하나만 곡마다 잔여 스캔까지 포함해 out/phase0/<곡>/final_<조합>/ 에 만든다.
- 곡마다 compare.png, 전체 listening_sheet.md(한국어 청취표)를 쓴다.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from mrengine.audio_io import SR, load_audio
from mrengine.ensemble import ensemble
from mrengine.pipeline import PipelineConfig, run_pipeline
from mrengine.separate import DEFAULT_MODELS, MODEL_DIR, AudioSeparatorBackend, Stems
from mrengine.spectral import N_FFT, frame_times, stft

AUDIO_EXTS = {".mp3", ".wav", ".flac"}
COMPARE_PAD_S = 5.0
MAX_LISTEN_SECTIONS = 3


class _NameOnly:
    """캐시된 분리 결과를 쓸 때 run_pipeline에 모델 이름만 넘기기 위한 자리표시."""

    def __init__(self, name: str):
        self.name = name

    def separate(self, mix):  # pragma: no cover - 잔여 스캔을 끈 실행에서는 불리지 않는다
        raise RuntimeError(f"{self.name}: 캐시 전용 분리기는 분리를 하지 않는다")


def combo_name(strength: float, method: str) -> str:
    return f"s{strength:.1f}_{method}"


def _save_atomic(path: Path, x: np.ndarray) -> None:
    """중간에 죽어도 반쯤 쓴 .npy가 캐시로 읽히지 않도록 임시 파일에 쓴 뒤 이름을 바꾼다."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, x.astype(np.float32))
    tmp.replace(path)


def _free_memory() -> None:
    """곡 사이에 큰 배열과 GPU 캐시를 놓아 준다 (3곡 연속 실행에서 메모리 부족으로 중단된 적 있음)."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_or_separate(mix: np.ndarray, song_dir: Path, models: list[str]) -> tuple[list[Stems], float]:
    """모델별 분리 결과를 캐시에서 읽거나, 없으면 분리해서 저장한다. -> (stems, 분리에 쓴 초)"""
    stems: list[Stems] = []
    spent = 0.0
    for m in models:
        d = song_dir / "stems_cache" / m
        fv, fi = d / "vocals.npy", d / "instrumental.npy"
        if fv.exists() and fi.exists():
            cached = Stems(vocals=np.load(fv), instrumental=np.load(fi))
            # 원곡이 바뀌었거나 저장이 깨진 캐시는 쓰지 않고 다시 분리한다
            if cached.vocals.shape == cached.instrumental.shape == mix.shape:
                stems.append(cached)
                continue
        t0 = time.perf_counter()
        sep = AudioSeparatorBackend(m, MODEL_DIR, song_dir / "_work")
        st = sep.separate(mix)
        del sep
        spent += time.perf_counter() - t0
        d.mkdir(parents=True, exist_ok=True)
        _save_atomic(fv, st.vocals)
        _save_atomic(fi, st.instrumental)
        stems.append(st)
    return stems, spent


def _choir_scores(report: dict, vocals: np.ndarray) -> list[tuple[float, int]]:
    """choir 구간마다 (점수, 인덱스). 점수 = 구간 보컬 에너지 합(크고 긴 합창일수록 높다)."""
    out = []
    for i, s in enumerate(report["sections"]):
        if s["kind"] != "choir":
            continue
        a, b = int(s["start_s"] * SR), int(s["end_s"] * SR)
        out.append((float(np.sum(np.square(vocals[:, a:b], dtype=np.float64))), i))
    return sorted(out, reverse=True)


def _find_gaudio(songs_dir: Path, stem: str) -> Path | None:
    g = songs_dir / "gaudio"
    if not g.is_dir():
        return None
    hits = sorted(p for p in g.iterdir() if p.suffix.lower() in AUDIO_EXTS and p.name.startswith(stem))
    return hits[0] if hits else None


def _spec_db(x: np.ndarray, a: int, b: int) -> np.ndarray:
    seg = x[:, a:b].mean(axis=0, keepdims=True)
    mag = np.abs(stft(seg))[0]
    return 20 * np.log10(mag + 1e-7)


def write_compare_png(path: Path, rows: list[tuple[str, np.ndarray]], report: dict, focus: dict | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fam in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(f.name == fam for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False

    dur = report["duration_s"]
    if focus is None:
        t0, t1 = 0.0, min(dur, 20.0)
    else:
        t0, t1 = max(0.0, focus["start_s"] - COMPARE_PAD_S), min(dur, focus["end_s"] + COMPARE_PAD_S)
    a, b = int(t0 * SR), int(t1 * SR)
    specs = [(name, _spec_db(x, a, b)) for name, x in rows]
    vmax = max(s.max() for _, s in specs)
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    fsel = freqs <= 12000

    fig, axes = plt.subplots(len(specs), 1, figsize=(12, 2.6 * len(specs)), sharex=True, squeeze=False)
    for ax, (name, s) in zip(axes[:, 0], specs):
        t = t0 + frame_times(s.shape[1])
        ax.pcolormesh(t, freqs[fsel] / 1000, s[fsel], vmin=vmax - 80, vmax=vmax, cmap="magma", shading="auto")
        for sec in report["sections"]:
            if sec["kind"] == "choir" and sec["end_s"] > t0 and sec["start_s"] < t1:
                a_, b_ = max(sec["start_s"], t0), min(sec["end_s"], t1)
                ax.axvspan(a_, b_, ymin=0.93, ymax=1.0, color="cyan", alpha=0.9, lw=0)  # 위쪽 띠
                ax.axvspan(a_, b_, color="cyan", alpha=0.08, lw=0)
                for x in (sec["start_s"], sec["end_s"]):
                    if t0 < x < t1:
                        ax.axvline(x, color="cyan", lw=1.0, ls="--")
        ax.set_ylabel("kHz")
        ax.set_title(name, loc="left", fontsize=10)
    axes[-1, 0].set_xlabel("시각 (초)  — 하늘색 띠와 점선 사이: choir 구간")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def process_song(song: Path, songs_dir: Path, out_root: Path, models: list[str], combos, final, cache_only,
                 reuse=False) -> dict:
    stem = song.stem
    song_dir = out_root / stem
    song_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    t_song = time.perf_counter()

    mix = load_audio(song)
    stems, timing["separate"] = load_or_separate(mix, song_dir, models)
    if cache_only:
        return {"song": str(song), "stem": stem, "timing_s": timing}

    names = [_NameOnly(m) for m in models]
    reports = {}
    t0 = time.perf_counter()
    for strength, method in combos:
        d = song_dir / combo_name(strength, method)
        if reuse and (d / "report.json").exists():
            rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
        else:
            cfg = PipelineConfig(strength=strength, ensemble_method=method, residual=False, write_mp3=False)
            rep = run_pipeline(song, d, names, cfg, stems=stems)
        reports[combo_name(strength, method)] = rep
    timing["sweep"] = time.perf_counter() - t0

    # 최종 조합: 잔여 스캔까지 포함. 첫 모델만 실제로 로드한다 (잔여 스캔이 separators[0]을 쓴다).
    t0 = time.perf_counter()
    fs, fm = final
    final_dir = song_dir / f"final_{combo_name(fs, fm)}"
    if reuse and (final_dir / "report.json").exists():
        final_rep = json.loads((final_dir / "report.json").read_text(encoding="utf-8"))
    else:
        seps = [AudioSeparatorBackend(models[0], MODEL_DIR, song_dir / "_work")] + names[1:]
        final_rep = run_pipeline(song, final_dir, seps, PipelineConfig(strength=fs, ensemble_method=fm), stems=stems)
        del seps
    timing["final"] = time.perf_counter() - t0

    vocals = ensemble([s.vocals for s in stems], mix.shape[1], fm)
    ranked = _choir_scores(final_rep, vocals)
    picks = [final_rep["sections"][i] | {"restore_amount": final_rep["restore_amount"][i],
                                         "gain_db": final_rep["section_gains_db"][i]} for _, i in ranked[:MAX_LISTEN_SECTIONS]]

    base_wav, rest_wav = load_audio(final_dir / "mr_baseline.wav"), load_audio(final_dir / "mr_restored.wav")
    for pk in picks:  # 사용자가 실제로 듣는 차이: restored가 baseline보다 몇 dB 큰가 (구간 전체 에너지)
        a, b = int(pk["start_s"] * SR), int(pk["end_s"] * SR)
        e_b = float(np.sum(np.square(base_wav[:, a:b], dtype=np.float64)))
        e_r = float(np.sum(np.square(rest_wav[:, a:b], dtype=np.float64)))
        pk["restored_vs_baseline_db"] = float(10 * np.log10((e_r + 1e-12) / (e_b + 1e-12)))
    rows = [("원곡", mix), ("baseline (앙상블만)", base_wav),
            (f"restored ({combo_name(fs, fm)}, 잔여 스캔 포함)", rest_wav)]
    gaudio = _find_gaudio(songs_dir, stem)
    if gaudio is not None:
        rows.append((f"Gaudio ({gaudio.name})", load_audio(gaudio)))
    write_compare_png(song_dir / "compare.png", rows, final_rep, picks[0] if picks else None)

    timing["total"] = time.perf_counter() - t_song
    secs = final_rep["sections"]
    summary = {
        "song": str(song.resolve()),
        "stem": stem,
        "duration_s": final_rep["duration_s"],
        "channels": final_rep["channels"],
        "n_sections": len(secs),
        "n_choir": sum(1 for s in secs if s["kind"] == "choir"),
        "choir_s": sum(s["end_s"] - s["start_s"] for s in secs if s["kind"] == "choir"),
        "final": combo_name(fs, fm),
        "final_dir": str(final_dir.resolve()),
        "compare_png": str((song_dir / "compare.png").resolve()),
        "gaudio": str(gaudio.resolve()) if gaudio else None,
        "picks": picks,
        "combo_restore": {k: _mean_choir(r) for k, r in reports.items()},
        "combo_dirs": {k: str((song_dir / k).resolve()) for k in reports},
        "final_timings_s": final_rep["timings_s"],
        "residual_spans": len(final_rep["residual_spans"]),
        "timing_s": timing,
    }
    (song_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _mean_choir(rep: dict) -> float:
    """choir 구간 길이로 가중한 평균 복구량."""
    w = [(s["end_s"] - s["start_s"], a) for s, a in zip(rep["sections"], rep["restore_amount"]) if s["kind"] == "choir"]
    tot = sum(x for x, _ in w)
    return float(sum(x * a for x, a in w) / tot) if tot else 0.0


def write_listening_sheet(path: Path, summaries: list[dict], songs_dir: Path) -> None:
    L = ["# Phase 0 청취 판정표", "",
         "합창 구간에서 반주가 **Gaudio Studio보다 확실히 더 꽉 차게** 들리면 통과입니다.",
         "",
         "## 먼저 알아 둘 점 (측정 결과)",
         "- 엔진의 '빈 주파수 채우기'(스펙트럼 복구)는 지금 모델 조합에서 거의 효과가 없습니다. 강도를 최대(1.0)로 해도",
         "  합창 구간 반주가 대부분 1dB도 안 커집니다. 주 분리 모델(BS-RoFormer)의 반주가 3곡 중 2곡(#13, 낡은 침대)에서",
         "  사실상 '원곡 - 보컬'이고(유봉은 약 -12dB 차이로 덜하지만 채울 여지는 역시 1dB 안팎),",
         "  원곡에서 보컬을 뺀 재료로는 더 채울 것이 거의 남지 않습니다.",
         "- 그래서 baseline과 restored의 차이는 대부분 아래 두 가지에서 나옵니다. 'restored-baseline' 칸이 그 구간의 실제 차이입니다.",
         "  1. 합창 구간에서 반주 음량을 **원곡의 음량 흐름에 맞춰 최대 +6dB 키움**. 원곡 음량에는 합창이 들어 있으므로",
         "     **원곡보다 반주가 더 커질 수도 있고**, 반주에 남은 보컬 찌꺼기도 같이 커질 수 있습니다.",
         "  2. MR에 남은 보컬 찌꺼기를 찾아 줄임.",
         "  (원곡 기반 스펙트럼 채우기는 이 모델 조합에서는 거의 0dB입니다.)",
         "- **\"더 크게 들리는 것\"과 \"더 꽉 차게 들리는 것\"을 구분해서 판단해 주세요.** restored가 그냥 더 큰 것뿐이라면",
         "  음량을 맞춰 들었을 때 차이가 사라집니다. 표의 '음량 보정' 칸이 그 구간에서 반주를 키운 양입니다.",
         "- 따라서 이번 청취의 핵심 질문은 \"restored가 Gaudio보다 꽉 차게 들리는가\"이고,",
         "  아니라면 채우는 방식 자체를 바꿔야 한다는 뜻입니다 (엔진을 계속 다듬기 전에 방향을 같이 정합니다).",
         "",
         "## 준비",
         f"- Gaudio Studio로 같은 곡의 MR을 뽑아 `{(songs_dir / 'gaudio').resolve()}` 폴더에 넣어 주세요.",
         "  파일 이름은 원곡 파일 이름으로 시작하면 됩니다 (예: `14. 낡은 침대를 타고 (Gaudio).mp3`).",
         "- 넣은 뒤 스크립트를 `--reuse`로 다시 돌리면(몇 분) 비교 그림(compare.png)에 Gaudio 줄이 추가됩니다.",
         "",
         "## 듣는 방법",
         "1. 아래 표의 **시각**으로 이동해 원곡 → baseline → restored → Gaudio 순서로 같은 부분을 들어 보세요.",
         "2. baseline은 분리 모델 결과 그대로, restored는 (1) 합창 구간 반주를 원곡 음량 흐름에 맞춰 키우고(원곡보다 커질 수 있음), (2) 남은 보컬 찌꺼기를 줄인 결과입니다.",
         "3. 이어폰/헤드폰으로, 음량을 같게 맞추고 들어 주세요.",
         "4. 표의 숫자 읽는 법:",
         "   - '스펙트럼 복구량': 빈 주파수를 원곡에서 다시 채운 양(원곡 에너지 대비 %). 0에 가까우면 이 단계는 거의 아무것도 안 한 것입니다.",
         "   - '음량 보정': 그 구간에서 restored의 반주를 키운 양(최대 +6dB). 원곡의 음량 흐름을 따르므로 원곡보다 커질 수 있습니다.",
         "   - 'restored-baseline': 두 파일을 그 구간에서 비교한 실제 음량 차이. 0dB 근처면 둘이 거의 같게 들립니다.",
         "   - '실제로 합창인가': 엔진이 '합창'이라고 표시한 구간이 정말 여러 명이 부르는 부분인지 알려 주세요 (검출 조정에 씁니다).",
         ""]
    for s in summaries:
        L += [f"## {s['stem']}", "",
              f"- 원곡: `{s['song']}`",
              f"- 길이 {mmss(s['duration_s'])}, {'모노' if s['channels'] == 1 else '스테레오'}, "
              f"합창 구간 {s['n_choir']}개 (총 {s['choir_s']:.0f}초) / 전체 보컬 구간 {s['n_sections']}개",
              f"- **추천: 먼저 `{s['final']}` 를 들어 보세요** (잔여 보컬 제거까지 한 최종본)",
              f"  - baseline: `{s['final_dir']}\\mr_baseline.mp3`",
              f"  - restored: `{s['final_dir']}\\mr_restored.mp3`",
              f"- Gaudio: {'`' + s['gaudio'] + '`' if s['gaudio'] else '(아직 없음)'}",
              f"- 비교 그림: `{s['compare_png']}`",
              "",
              "| 시각 | 길이 | 스펙트럼 복구량 | 음량 보정 | restored-baseline | 실제로 합창인가 | Gaudio보다 꽉 차는가 | 합창이 새는가 | 이상한 소리가 나는가 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for p in s["picks"]:
            L.append(f"| {mmss(p['start_s'])} ~ {mmss(p['end_s'])} (반주 +{p['gain_db']:.1f}dB) | {p['end_s'] - p['start_s']:.1f}초 | "
                     f"{p['restore_amount'] * 100:.1f}% | +{p['gain_db']:.1f}dB | {p['restored_vs_baseline_db']:+.1f}dB | "
                     "[ ] 예 [ ] 아니오 | [ ] 예 [ ] 아니오 | [ ] 예 [ ] 아니오 | [ ] 예 [ ] 아니오 |")
        if not s["picks"]:
            L.append("| (합창 구간이 검출되지 않음) | | | | | | | | |")
        L += ["", "다른 조합 (잔여 보컬 제거 없이, 비교용 — `mr_restored.wav`):", "",
              "| 조합 | 합창 구간 평균 복구량 | 폴더 |", "|---|---|---|"]
        for k, v in s["combo_restore"].items():
            L.append(f"| {k} | {v * 100:.2f}% | `{s['combo_dirs'][k]}` |")
        L += ["", "메모: ____________________________________________", ""]
    L += ["## 조합 이름 읽는 법",
          "- `s0.5` 같은 숫자: 복구 강도(0이면 baseline과 같음, 1이면 최대로 채움).",
          "- `avg`: 두 모델 결과를 평균. `min_mag`: 두 모델 중 더 작은 쪽(보컬이 덜 새지만 반주도 얇아질 수 있음).",
          ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 0 일괄 비교")
    p.add_argument("--songs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--strengths", default="0.3,0.5,0.8,1.0")
    p.add_argument("--ensembles", default="avg,min_mag")
    p.add_argument("--final", default="1.0,avg", help="잔여 스캔까지 돌릴 조합 'strength,ensemble'")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--only", default="", help="이 문자열이 이름에 든 곡만")
    p.add_argument("--cache-only", action="store_true", help="분리 캐시만 만들고 끝낸다")
    p.add_argument("--reuse", action="store_true",
                   help="이미 있는 조합 결과(report.json)는 다시 돌리지 않는다. Gaudio 파일을 넣은 뒤 그림·청취표만 다시 만들 때")
    args = p.parse_args(argv)

    songs = sorted(x for x in args.songs.iterdir() if x.is_file() and x.suffix.lower() in AUDIO_EXTS)
    if args.only:
        songs = [s for s in songs if args.only in s.name]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    combos = [(float(s), e.strip()) for e in args.ensembles.split(",") for s in args.strengths.split(",")]
    fs, fm = args.final.split(",")
    final = (float(fs), fm.strip())

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for song in songs:
        print(f"== {song.name}", flush=True)
        s = process_song(song, args.songs, args.out, models, combos, final, args.cache_only, args.reuse)
        print(f"   {json.dumps(s['timing_s'])}", flush=True)
        summaries.append(s)
        _free_memory()
    if not args.cache_only:
        write_listening_sheet(args.out / "listening_sheet.md", summaries, args.songs)
        print(f"청취표: {(args.out / 'listening_sheet.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
