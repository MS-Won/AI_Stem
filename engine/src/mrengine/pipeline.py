"""전체 흐름: 분리 -> 앙상블 -> 구간 검출 -> 합창 복구 -> 음량 복구 -> 잔여 스캔 -> 저장."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mrengine import loudness as loudness_mod
from mrengine import residual as residual_mod
from mrengine import restore as restore_mod
from mrengine import sections as sections_mod
from mrengine import separate as separate_mod
from mrengine.audio_io import SR, load_audio, write_mp3, write_wav
from mrengine.ensemble import EnsembleMethod, ensemble
from mrengine.loudness import finalize, restore_section_loudness
from mrengine.residual import scan_and_reduce
from mrengine.restore import restore_choir
from mrengine.sections import detect_vocal_sections
from mrengine.separate import Separator, Stems
from mrengine.spectral import frame_times


@dataclass
class PipelineConfig:
    strength: float = 0.5
    ensemble_method: EnsembleMethod = "avg"
    max_gain_db: float = 6.0
    residual_threshold_db: float = -30.0
    write_mp3: bool = True
    residual: bool = True  # False면 잔여 보컬 스캔(재분리, 가장 느림)을 건너뛴다. Phase 0 조합 탐색용


def separate_all(mix: np.ndarray, separators: list[Separator]) -> list[Stems]:
    """분리기마다 한 번씩 분리한다. 결과를 캐시해 run_pipeline(stems=...)에 재사용할 수 있다."""
    return [s.separate(mix) for s in separators]


def finalize_pair(baseline: np.ndarray, restored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """두 MR을 같은 배율로 피크 처리한다. 따로 정규화하면 음량 보정으로 피크가 커진 restored만 통째로
    작아져(실제 곡에서 최대 -6dB) 같은 시각의 A/B 비교가 음량 차이에 묻힌다."""
    both = finalize(np.concatenate([baseline, restored], axis=0))
    return both[: baseline.shape[0]], both[baseline.shape[0] :]


def _constants() -> dict:
    """리포트를 재현할 수 있도록 실제 쓰인 모듈 상수를 기록한다."""
    def pick(mod, names):
        return {n: getattr(mod, n) for n in names}

    return {
        "sections": pick(sections_mod, ["V_RATIO_THRESHOLD", "CHOIR_ABS_WIDTH_HZ", "CHOIR_WIDTH_RATIO",
                                        "REF_WIDTH_PERCENTILE", "CHOIR_FLATNESS_THRESHOLD", "BAND_LO_HZ",
                                        "BAND_HI_HZ", "ENERGY_FRACTION", "MEDIAN_FILTER_S",
                                        "CHOIR_MEDIAN_FILTER_S"]),
        "restore": pick(restore_mod, ["EDGE_RAMP_S", "MIN_REF_FRAME_FRACTION", "PHASE_SWITCH"]),
        "loudness": {**pick(loudness_mod, ["REF_WINDOW_S", "MIN_REF_S", "RAMP_S"]), "ceiling_db": -1.0},
        "residual": {**pick(residual_mod, ["WINDOW_S", "FADE_S"]), "reduction": 0.8},
        "separate": pick(separate_mod, ["INPUT_HEADROOM"]),
    }


def _section_amounts(sections, amount: np.ndarray) -> list[float]:
    t = frame_times(len(amount))
    out = []
    for sec in sections:
        sel = (t >= sec.start_s) & (t < sec.end_s)
        out.append(float(amount[sel].mean()) if sec.kind == "choir" and sel.any() else 0.0)
    return out


def run_pipeline(
    input_path: Path,
    out_dir: Path,
    separators: list[Separator],
    config: PipelineConfig | None = None,
    stems: list[Stems] | None = None,
) -> dict:
    config = config or PipelineConfig()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t0
        now = time.perf_counter()
        timings[name] = now - t0
        t0 = now

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mix = load_audio(input_path)
    n = mix.shape[1]
    lap("load")

    if stems is None:
        stems = separate_all(mix, separators)
        lap("separate")
    else:
        if len(stems) != len(separators):
            raise ValueError("stems와 separators 개수가 다르다")
        timings["separate"] = 0.0

    baseline = ensemble([s.instrumental for s in stems], n, config.ensemble_method)
    vocals = ensemble([s.vocals for s in stems], n, config.ensemble_method)
    lap("ensemble")

    sections = detect_vocal_sections(vocals, mix)
    lap("sections")

    restored, amount = restore_choir(mix, baseline, vocals, sections, config.strength)
    lap("restore")

    restored, gains = restore_section_loudness(restored, mix, sections, config.max_gain_db)
    lap("loudness")

    if config.residual:
        restored, spans = scan_and_reduce(restored, separators[0], config.residual_threshold_db)
    else:
        spans = []
    lap("residual")

    peak_before = max(float(np.max(np.abs(baseline))), float(np.max(np.abs(restored))))
    baseline, restored = finalize_pair(baseline, restored)
    peak_after = max(float(np.max(np.abs(baseline))), float(np.max(np.abs(restored))))
    final_gain_db = 20 * np.log10(peak_after / peak_before) if peak_before > 0 and peak_after > 0 else 0.0
    write_wav(out_dir / "mr_baseline.wav", baseline)
    write_wav(out_dir / "mr_restored.wav", restored)
    if config.write_mp3:
        write_mp3(out_dir / "mr_baseline.mp3", baseline)
        write_mp3(out_dir / "mr_restored.mp3", restored)
    lap("write")

    report = {
        "input": str(input_path),
        "duration_s": n / SR,
        "channels": int(mix.shape[0]),
        "models": [s.name for s in separators],
        "config": asdict(config),
        "sections": [s.to_dict() for s in sections],
        "section_gains_db": [float(g) for g in gains],
        "final_gain_db": float(final_gain_db),  # finalize_pair가 두 MR에 같이 곱한 감쇄(dB, 0 이하)
        "constants": _constants(),
        "restore_amount": _section_amounts(sections, amount),
        "residual_spans": [s.to_dict() for s in spans],
        "timings_s": timings,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return json.loads(json.dumps(report))
