"""분리 백엔드. audio-separator(UVR 모델)를 작은 프로토콜 뒤에 감싼다."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from mrengine.audio_io import SR, fit_length

# 모델 파일 저장 위치(gitignore). 모든 곳에서 이 경로 하나를 공유한다.
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# Phase 0 앙상블 후보. 선정 근거와 라이선스는 engine/MODELS.md.
DEFAULT_MODELS: list[str] = [
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",  # RoFormer 보컬/반주
    "htdemucs_ft.yaml",  # Demucs 4-stem
]

INPUT_HEADROOM = 0.9  # 분리기에 넣는 입력의 최대 피크. 라이브러리의 피크 1.0 정규화를 피한다

_STEM_RE = re.compile(r"_\(([^)]+)\)_")


@dataclass
class Stems:
    vocals: np.ndarray
    instrumental: np.ndarray


class Separator(Protocol):
    name: str

    def separate(self, mix: np.ndarray) -> Stems: ...


def _match_channels(x: np.ndarray, channels: int) -> np.ndarray:
    if x.shape[0] == channels:
        return x
    if channels == 1:
        return x.mean(axis=0, keepdims=True)
    return np.repeat(x[:1], channels, axis=0) if x.shape[0] == 1 else x[:channels]


def normalize_stems(vocals: np.ndarray, instrumental: np.ndarray, like: np.ndarray) -> Stems:
    """모델 출력의 채널 수와 길이를 입력(`like`)과 똑같이 맞춘다."""
    def fix(x: np.ndarray) -> np.ndarray:
        x = _match_channels(np.asarray(x, dtype=np.float32), like.shape[0])
        return np.ascontiguousarray(fit_length(x, like.shape[1]), dtype=np.float32)

    return Stems(vocals=fix(vocals), instrumental=fix(instrumental))


class AudioSeparatorBackend:
    """audio-separator 모델 하나. 모델은 생성 시 한 번 로드해 재사용한다."""

    def __init__(self, model_filename: str, model_dir: Path, work_dir: Path, _sep=None):
        """`_sep`은 테스트용: 라이브러리 Separator 대신 쓸 객체(load_model 불필요)."""
        self.name = model_filename
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        if _sep is not None:
            self._sep = _sep
            return
        from audio_separator.separator import Separator as _Separator

        self._sep = _Separator(
            model_file_dir=str(model_dir),
            output_dir=str(self.work_dir),
            output_format="WAV",
            normalization_threshold=1.0,  # 피크가 1을 넘을 때만 줄인다. 상대 크기를 건드리지 않기 위해
            sample_rate=SR,
        )
        self._sep.load_model(model_filename)

    def separate(self, mix: np.ndarray) -> Stems:
        # 경로의 한글/공백/'#'이 문제되지 않도록 ASCII 임시 이름으로 쓴다.
        inp = self.work_dir / "input.wav"
        # audio-separator(MDXC)는 입력 피크가 1을 넘으면 1로 줄여 분리하고 되돌리지 않는다. 음량 보정된 MR처럼
        # 피크가 1을 넘는 입력은 보컬이 1/피크배로 작게 나와 잔여 스캔이 어긋난다. 미리 피크 0.9 이하로 낮춰
        # 라이브러리가 건드리지 않게 하고, 나온 스템에 같은 배율을 곱해 원래 크기로 돌린다.
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        p = max(1.0, peak / INPUT_HEADROOM)
        sf.write(str(inp), (mix / p).T, SR, subtype="FLOAT")
        try:
            outs = self._sep.separate(str(inp))
        finally:
            inp.unlink(missing_ok=True)

        stems: dict[str, np.ndarray] = {}
        for out in outs:
            path = Path(out)
            if not path.is_absolute():
                path = self.work_dir / path
            m = _STEM_RE.search(path.name)
            if not m:
                continue
            data, _ = sf.read(str(path), dtype="float32", always_2d=True)
            stems[m.group(1).lower()] = data.T
            path.unlink(missing_ok=True)

        if "vocals" not in stems:
            raise RuntimeError(f"{self.name}: 보컬 스템이 없다. 출력={sorted(stems)}")
        if "instrumental" in stems:
            inst = stems["instrumental"]
        else:  # 4-stem 모델: 비보컬 스템 합
            others = [v for k, v in stems.items() if k != "vocals"]
            if not others:
                raise RuntimeError(f"{self.name}: 반주 스템이 없다.")
            n = min(o.shape[1] for o in others)
            inst = sum(o[:, :n] for o in others)
        return normalize_stems(stems["vocals"] * p, inst * p, like=mix)
