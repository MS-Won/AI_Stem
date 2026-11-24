# Phase 0 — 합창 복구 MR 엔진 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 로컬 RTX 3060에서 곡 하나를 넣으면 `mr_baseline.wav`(앙상블 MR)과 `mr_restored.wav`(합창 구간 복구 MR), `report.json`을 만들어 주는 Python 엔진과, 테스트 3곡을 일괄 처리해 Gaudio 결과와 비교 청취할 수 있게 하는 스크립트.

**Architecture:** `engine/` 아래 독립 Python 패키지 `mrengine`. 분리는 `audio-separator` 패키지(UVR 모델 래퍼)로 하고, 앙상블·합창 검출·복구·음량 보정·잔여 보컬 처리는 numpy STFT 위에서 직접 구현한다. 분리 백엔드는 주입 가능하게 해서 단위 테스트는 합성 신호로 GPU 없이 돈다.

**Tech Stack:** Python 3.12, numpy, scipy, soundfile, audio-separator[gpu] (PyTorch CUDA), matplotlib(비교 스펙트로그램), pytest, ffmpeg(MP3 인코딩, 시스템에 설치됨).

**Spec:** `docs/superpowers/specs/2026-11-24-choir-mr-mvp-design.md` (§5 엔진, §7 Phase 0, §8 검증)

## Global Constraints

- Python 3.12, 가상환경은 `engine/.venv` (git 제외).
- 샘플레이트 44100Hz. **모노 입력을 받고, 출력 채널 수는 입력을 따른다.**
- 오디오 배열 형식은 엔진 전체에서 `np.ndarray`, dtype `float32`, shape `(channels, samples)`.
- 출력 길이는 입력 길이와 정확히 같다 (모델 출력이 길거나 짧으면 자르거나 0으로 채움).
- **복구는 원곡 믹스에 있는 정보만 쓴다. 생성 모델로 채우지 않는다 (반목표 A1).** 복구 후 어떤 시간-주파수 칸에서도 MR 크기가 원곡 믹스 크기를 넘지 않는다.
- 출력은 WAV 16-bit PCM 44.1kHz + MP3 320kbps. 최종 피크는 -1 dBFS 이하.
- `Test AR/`의 음원과 모든 처리 결과물(`engine/out/`), 모델 파일(`engine/models/`)은 커밋하지 않는다 (`.gitignore`에 이미 있음).
- 모델은 상업적 이용이 가능한 라이선스인지 Task 3에서 확인하고 `engine/MODELS.md`에 기록한다.
- 외부 라이브러리 API는 이 계획서의 예시를 믿지 말고 설치된 버전의 실제 시그니처를 확인한다 (Shorts 프로젝트의 교훈).
- 커밋 메시지는 한국어로, 왜 그렇게 했는지를 남긴다. 끝에 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- 테스트 명령은 `engine/`에서 `.venv/Scripts/python -m pytest` (Windows, Git Bash 기준).

## Review Focus

1. **합창 구간이 없는 곡** — 검출 구간이 0개여도 파이프라인이 죽지 않고 `mr_restored`가 `mr_baseline`과 (음량 안전 처리 외에는) 같아야 한다. → Task 8 `test_pipeline_no_vocal_sections_returns_baseline`
2. **모노 입력** (`#13 Do You Hear the People Sing.mp3`가 모노) — 분리 모델이 스테레오로 돌려줘도 출력은 모노 1채널. → Task 1 `test_load_mono_keeps_one_channel`, Task 3 `test_separate_restores_input_channel_count`
3. **파일명에 `#`, 공백, 한글** — 경로 처리와 `audio-separator` 출력 파일 찾기가 깨지기 쉽다. → Task 3 `test_separate_handles_unicode_and_hash_path`
4. **음량 보정 후 클리핑** — 게인을 올린 구간에서 피크가 0 dBFS를 넘으면 안 된다. → Task 6 `test_finalize_peak_never_exceeds_minus_1dbfs`
5. **모델 출력 길이 불일치** — 앙상블 시 모델마다 길이가 몇 샘플씩 달라 broadcast 에러가 난다. → Task 2 `test_ensemble_aligns_different_lengths`

---

## File Structure

```
engine/
  pyproject.toml            패키지 메타데이터, 의존성, pytest 설정
  MODELS.md                 사용 모델, 출처, 라이선스, 선택 근거 (Task 3, Task 9)
  src/mrengine/
    __init__.py
    audio_io.py             load_audio / write_wav / write_mp3 / fit_length
    spectral.py             stft / istft 공통 파라미터
    ensemble.py             여러 MR을 STFT 영역에서 합치기
    separate.py             분리 백엔드 프로토콜 + audio-separator 구현
    sections.py             Section 타입 + 보컬/합창 구간 검출
    restore.py              합창 구간 반주 복구 (spectral hole 채우기)
    loudness.py             구간별 음량 복구 + 최종 피크 안전 처리
    residual.py             잔여 보컬 스캔·감쇄
    pipeline.py             전체 흐름 + report.json
    cli.py                  python -m mrengine.cli
  scripts/
    phase0_batch.py         Test AR 일괄 처리 + 비교 스펙트로그램 + 청취 시트
  tests/
    conftest.py             합성 신호 fixture
    test_*.py               모듈별 1:1
CLAUDE.md                   작업 재개 가이드 (저장소 루트)
docs/STATE.md               현재 상태와 다음 할 일
todo.md                     작업 큐
```

---

### Task 1: 엔진 뼈대, 작업 문서, 오디오 입출력

**Files:**
- Create: `engine/pyproject.toml`, `engine/src/mrengine/__init__.py`, `engine/src/mrengine/audio_io.py`, `engine/tests/conftest.py`, `engine/tests/test_audio_io.py`, `CLAUDE.md`, `docs/STATE.md`, `todo.md`

**Interfaces:**
- Produces:
  - `SR: int = 44100`
  - `load_audio(path: str | Path) -> np.ndarray` — `(channels, samples)` float32, 44100Hz로 리샘플(필요 시 ffmpeg로 디코딩). 채널 수 보존.
  - `fit_length(audio: np.ndarray, n_samples: int) -> np.ndarray` — 자르거나 끝에 0 채움.
  - `write_wav(path, audio: np.ndarray) -> None` — 16-bit PCM.
  - `write_mp3(path, audio: np.ndarray) -> None` — ffmpeg로 320kbps.
  - `conftest.py` fixture: `tone(freq_hz, seconds, amp) -> np.ndarray (1, n)`, `band_noise(lo_hz, hi_hz, seconds, amp, seed) -> np.ndarray (1, n)`.

- [ ] **Step 1: 뼈대와 문서**
  - `pyproject.toml`: 패키지명 `mrengine`, `src/` 레이아웃, 의존성 `numpy`, `scipy`, `soundfile`, `matplotlib`; optional `gpu = ["audio-separator[gpu]"]`; pytest `testpaths=["tests"]`, `pythonpath=["src"]`, marker `gpu: 실제 모델과 GPU가 필요한 테스트`.
  - `engine/.venv` 생성 후 `pip install -e .[gpu] pytest`. PyTorch가 CUDA를 잡는지 확인: `.venv/Scripts/python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True NVIDIA GeForce RTX 3060`. 안 잡히면 PyTorch 공식 CUDA 휠로 재설치하고 방법을 `CLAUDE.md`에 남긴다.
  - `CLAUDE.md`: 이전 프로젝트(Shorts)와 같은 형식. "세션 시작 시 `docs/STATE.md`부터 읽어라", 문서 지도(spec, plan, STATE, todo), 함정 목록(Global Constraints의 길이·모노·A1 원칙, Test AR 커밋 금지), 작업 규칙(pytest 통과 후 커밋, 한국어 커밋).
  - `docs/STATE.md`: 현재 Phase 0 Task 1 진행 중, 다음 할 일.
  - `todo.md`: 이 계획의 Task 1~9를 체크박스로.

- [ ] **Step 2: 실패하는 테스트 작성** (`tests/test_audio_io.py`)

```python
def test_load_mono_keeps_one_channel(tmp_path, tone):
    p = tmp_path / "mono.wav"; write_wav(p, tone(440, 1.0, 0.5))
    a = load_audio(p)
    assert a.shape[0] == 1 and a.dtype == np.float32

def test_load_resamples_to_44100(tmp_path):
    # 22050Hz 스테레오 1초 파일을 soundfile로 직접 써서
    a = load_audio(p)
    assert a.shape == (2, 44100)

def test_load_mp3_with_unicode_hash_path(tmp_path, tone):
    p = tmp_path / "#13 낡은 침대.mp3"; write_mp3(p, tone(440, 1.0, 0.5))
    a = load_audio(p)
    assert a.shape[0] == 1 and abs(a.shape[1] - 44100) < 2048  # MP3 인코더 패딩 허용

def test_fit_length_pads_and_trims(tone):
    x = tone(440, 1.0, 0.5)
    assert fit_length(x, 100).shape == (1, 100)
    y = fit_length(x, 50000)
    assert y.shape == (1, 50000) and np.all(y[:, 44100:] == 0)

def test_write_wav_is_16bit(tmp_path, tone):
    p = tmp_path / "o.wav"; write_wav(p, tone(440, 0.1, 0.5))
    assert sf.info(p).subtype == "PCM_16"
```

- [ ] **Step 3: 실패 확인** — `.venv/Scripts/python -m pytest tests/test_audio_io.py -v` → `ImportError`/FAIL.
- [ ] **Step 4: `audio_io.py` 구현** — WAV/FLAC는 soundfile, MP3 등은 `ffmpeg -i <in> -f f32le -ac <원래 채널> -ar 44100 -` 파이프로 디코딩(채널 수는 `ffprobe`로 읽음). 리샘플은 ffmpeg에 맡긴다. `write_mp3`는 임시 WAV → `ffmpeg -b:a 320k`.
- [ ] **Step 5: 통과 확인** — 같은 명령 → 5 passed.
- [ ] **Step 6: 커밋** — `엔진 뼈대와 오디오 입출력 추가` (왜: 모노 곡과 한글·# 경로를 처음부터 지원해야 테스트 3곡을 그대로 쓸 수 있음).

---

### Task 2: STFT 공통 함수와 앙상블

**Files:**
- Create: `engine/src/mrengine/spectral.py`, `engine/src/mrengine/ensemble.py`, `engine/tests/test_ensemble.py`

**Interfaces:**
- Consumes: `fit_length` (Task 1)
- Produces:
  - `N_FFT = 4096`, `HOP = 1024`
  - `stft(audio: np.ndarray) -> np.ndarray` — complex64, shape `(channels, freqs, frames)`, `scipy.signal.stft` Hann 창.
  - `istft(spec: np.ndarray, n_samples: int) -> np.ndarray` — `(channels, n_samples)` float32.
  - `EnsembleMethod = Literal["avg", "min_mag", "max_mag"]`
  - `ensemble(mrs: list[np.ndarray], n_samples: int, method: EnsembleMethod = "avg") -> np.ndarray` — 각 MR을 `fit_length`로 맞춘 뒤 STFT 영역에서 합침. `avg`는 복소 평균, `min_mag`/`max_mag`는 칸마다 크기가 가장 작은/큰 모델의 복소값을 그대로 선택.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_stft_roundtrip(tone):
    x = tone(440, 1.0, 0.5)
    assert np.allclose(istft(stft(x), x.shape[1]), x, atol=1e-4)

def test_ensemble_single_input_is_identity(tone):
    x = tone(440, 1.0, 0.5)
    assert np.allclose(ensemble([x], x.shape[1]), x, atol=1e-4)

def test_ensemble_aligns_different_lengths(tone):
    a = tone(440, 1.0, 0.5); b = a[:, :-37]
    assert ensemble([a, b], a.shape[1], "avg").shape == a.shape

def test_min_mag_picks_quieter_model(tone):
    a = tone(440, 1.0, 0.5); b = tone(440, 1.0, 0.1)
    out = ensemble([a, b], a.shape[1], "min_mag")
    assert np.isclose(rms(out), rms(b), rtol=0.05)

def test_max_mag_picks_louder_model(tone): ...  # 위와 대칭, rms(out) ≈ rms(a)
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_ensemble.py -v` → FAIL.
- [ ] **Step 3: 구현** — `spectral.py`, `ensemble.py`. 메모리를 위해 complex64 유지 (4분 30초 스테레오 한 장 ≈ 190MB).
- [ ] **Step 4: 통과 확인** — 5 passed.
- [ ] **Step 5: 커밋** — `STFT 공통 함수와 MR 앙상블 추가` (왜: 사용자가 Vocal Cut과 Stem 결과를 손으로 비교하던 작업을 자동화하는 기반).

---

### Task 3: 분리 백엔드

**Files:**
- Create: `engine/src/mrengine/separate.py`, `engine/tests/test_separate.py`, `engine/MODELS.md`

**Interfaces:**
- Consumes: `load_audio`, `write_wav`, `fit_length` (Task 1)
- Produces:
  - `@dataclass Stems: vocals: np.ndarray; instrumental: np.ndarray` — 둘 다 입력과 같은 shape.
  - `class Separator(Protocol): name: str; def separate(self, mix: np.ndarray) -> Stems`
  - `class AudioSeparatorBackend(Separator)` — `__init__(self, model_filename: str, model_dir: Path, work_dir: Path)`. 4-stem 모델이면 `instrumental = 비보컬 스템 합`.
  - `DEFAULT_MODELS: list[str]` — Phase 0에서 앙상블에 쓸 모델 파일명 (Step 1에서 확정).

- [ ] **Step 1: 모델 조사와 확정** — `.venv/Scripts/audio-separator --list_models`로 실제 이름을 확인하고 아래 성격의 모델을 고른다.
  - RoFormer 계열 보컬/반주 모델 1개 이상 (BS-RoFormer 또는 Mel-Band RoFormer, 목록에서 SDR이 높은 것)
  - Demucs 계열 4-stem 모델 1개 (`htdemucs_ft` 계열)
  - 가능하면 합창·백보컬을 보컬로 보지 않는 "karaoke" 계열 모델 1개 (비교용)
  - 각 모델의 출처·라이선스·상업적 이용 가능 여부를 `MODELS.md`에 표로 기록. 상업 이용 불가 모델은 Phase 0 비교용으로만 표시.
- [ ] **Step 2: 실패하는 테스트**

```python
class FakeBackend:  # 테스트 내부: 백엔드가 스테레오를 돌려주는 상황 흉내
    ...

def test_separate_restores_input_channel_count(tone):
    # 모노 입력 → AudioSeparatorBackend의 후처리 함수 normalize_stems()가
    # 스테레오 모델 출력을 평균내 모노로 되돌림
    stems = normalize_stems(raw_vocals_stereo, raw_inst_stereo, like=mono_mix)
    assert stems.vocals.shape == mono_mix.shape and stems.instrumental.shape == mono_mix.shape

def test_separate_trims_model_length(tone): ...  # 모델 출력이 길이가 다를 때 입력 길이로

@pytest.mark.gpu
def test_separate_handles_unicode_and_hash_path(tmp_path, tone):
    work = tmp_path / "작업 #1"
    be = AudioSeparatorBackend(DEFAULT_MODELS[0], MODEL_DIR, work)
    stems = be.separate(tone(440, 5.0, 0.5))
    assert stems.instrumental.shape == (1, 5 * 44100)

@pytest.mark.gpu
def test_real_model_separates_tone_from_noise(...):
    # 합성 신호로 결과가 나오는지만 확인 (품질 판정은 Task 9 청취)
```

  - `normalize_stems(vocals, instrumental, like: np.ndarray) -> Stems`를 `separate.py`의 공개 함수로 둔다.
- [ ] **Step 3: 실패 확인** — `pytest tests/test_separate.py -v` → FAIL.
- [ ] **Step 4: 구현** — 입력을 `work_dir`에 ASCII 임시 이름(`input.wav`)으로 써서 `audio-separator`에 넘기고, 반환된 출력 파일 목록에서 스템 이름(`Vocals`, `Instrumental`, `Drums`… — 실제 표기는 설치 버전에서 확인)으로 찾는다. 모델은 `__init__`에서 한 번 로드해 재사용.
- [ ] **Step 5: 통과 확인** — `pytest tests/test_separate.py -v` (gpu 포함) → 전부 PASS. CPU 전용 테스트는 `-m "not gpu"`로도 돈다.
- [ ] **Step 6: 커밋** — `audio-separator 분리 백엔드와 모델 라이선스 기록 추가`.

---

### Task 4: 보컬·합창 구간 검출

**Files:**
- Create: `engine/src/mrengine/sections.py`, `engine/tests/test_sections.py`

**Interfaces:**
- Consumes: `stft` (Task 2)
- Produces:
  - `@dataclass(frozen=True) Section: start_s: float; end_s: float; kind: Literal["choir", "lead"]` + `to_dict()`.
  - `detect_vocal_sections(vocals: np.ndarray, mix: np.ndarray, min_len_s: float = 1.0) -> list[Section]`
  - `frames_mask(sections, n_frames: int, kinds: set[str]) -> np.ndarray[bool]` — 프레임 단위 마스크 (Task 5, 7에서 사용).

**알고리즘 (테스트가 결정하지 못하는 부분):**
- 프레임별 보컬 에너지 비 `v_ratio = |V|² 합 / |M|² 합` (채널·주파수 합). `v_ratio > 0.25`인 프레임이 보컬 활성.
- 보컬 활성 프레임에서 보컬의 대역 폭 = 200Hz~8kHz 안에서 보컬 에너지의 80%가 들어가는 주파수 폭. 이 폭이 같은 곡의 보컬 활성 프레임 중앙값의 1.5배 이상이거나, 보컬 스펙트럼 평탄도(spectral flatness)가 0.3 이상이면 `choir`, 아니면 `lead`. (임계값은 모듈 상수로 두고 Task 9에서 실제 곡으로 조정)
- 0.5초 중앙값 필터로 프레임 라벨을 다듬고, `min_len_s`보다 짧은 구간은 버린다.

- [ ] **Step 1: 실패하는 테스트** — 합성 신호: 0~3초 악기(200Hz 톤), 3~6초 악기 + 넓은 대역 노이즈(choir 흉내, 300Hz~6kHz), 6~9초 악기 + 좁은 톤 묶음(lead 흉내, 440·880·1320Hz). `vocals`는 노이즈·톤 부분만.

```python
def test_detects_choir_section(synthetic_song):
    secs = detect_vocal_sections(v, mix)
    choir = [s for s in secs if s.kind == "choir"]
    assert len(choir) == 1 and abs(choir[0].start_s - 3) < 0.3 and abs(choir[0].end_s - 6) < 0.3

def test_detects_lead_section(synthetic_song): ...  # 6~9초가 lead

def test_no_vocals_gives_no_sections(tone):
    x = tone(200, 5.0, 0.5)
    assert detect_vocal_sections(np.zeros_like(x), x) == []

def test_short_blips_are_dropped(...):  # 0.3초 노이즈 → 구간 없음
```

- [ ] **Step 2: 실패 확인** → FAIL.
- [ ] **Step 3: 구현**.
- [ ] **Step 4: 통과 확인** → 4 passed.
- [ ] **Step 5: 커밋** — `합창·리드 보컬 구간 검출 추가`.

---

### Task 5: 합창 구간 반주 복구 (핵심 차별점)

**Files:**
- Create: `engine/src/mrengine/restore.py`, `engine/tests/test_restore.py`

**Interfaces:**
- Consumes: `stft`, `istft` (Task 2), `Section`, `frames_mask` (Task 4)
- Produces:
  - `restore_choir(mix: np.ndarray, inst: np.ndarray, vocals: np.ndarray, sections: list[Section], strength: float = 0.5) -> tuple[np.ndarray, np.ndarray]` — `(복구된 inst, 복구량 맵 (frames,) float32)`. 복구량 맵은 프레임별로 더해진 에너지 비율 (report와 UI용).

**알고리즘 (스펙 §5 [4]):**
```
M, I, V = stft(mix), stft(inst), stft(vocals)
R = M - V                                   # 원곡에서 보컬 추정치를 뺀 나머지 (복구의 유일한 재료)
ref = 보컬이 없는 프레임(sections 밖)에서 median(|I| / (|M|+eps)), 주파수별   # 이 곡의 '정상 반주 비율'
   (보컬 없는 프레임이 전체의 10% 미만이면 ref = 곡 전체 median으로 대체)
choir 프레임에서만:
  target = min(ref[f] * |M|, |R|)            # 정상 비율만큼, 그러나 원곡-보컬 이상은 절대 안 넘음
  deficit = max(target - |I|, 0)
  new_mag = |I| + strength * deficit
  phase  = |I|가 target의 25% 미만인 칸은 R의 위상, 아니면 I의 위상
  I'[f,t] = new_mag * phase
경계 0.1초는 strength를 선형으로 올리고 내려서 클릭 방지
마지막으로 |I'| <= |M| 클램프 (Global Constraint)
```

- [ ] **Step 1: 실패하는 테스트** — 합성: 악기 = 1kHz 톤(전 구간), 합창 = 300Hz~6kHz 노이즈(3~6초). `inst` 추정치 = 악기에서 3~6초 동안 1kHz 성분을 0.1배로 깎은 것(spectral hole 흉내).

```python
def test_strength_zero_is_identity(...):
    out, _ = restore_choir(mix, inst, vocals, secs, strength=0.0)
    assert np.allclose(out, inst, atol=1e-4)

def test_fills_hole_in_choir_section(...):
    out, _ = restore_choir(mix, inst, vocals, secs, strength=1.0)
    # 3~6초 구간 1kHz 대역 에너지가 원래 악기의 70% 이상으로 돌아옴
    assert band_rms(out, 3, 6, 1000) >= 0.7 * band_rms(instrument, 3, 6, 1000)

def test_does_not_touch_outside_sections(...):
    # 0~2.9초 구간은 strength=1.0에서도 inst와 동일
def test_never_exceeds_mix_magnitude(...):
    assert np.all(np.abs(stft(out)) <= np.abs(stft(mix)) + 1e-3)
def test_no_choir_sections_returns_input(...):
def test_restore_amount_map_is_positive_only_in_choir(...):
```

- [ ] **Step 2: 실패 확인** → FAIL.
- [ ] **Step 3: 구현**.
- [ ] **Step 4: 통과 확인** → 6 passed.
- [ ] **Step 5: 커밋** — `합창 구간 반주 복구 추가` (왜: 원곡 믹스에 남은 악기 성분만으로 채워서 지어낸 소리 없이 '꽉 찬 반주'를 만들기 위함).

---

### Task 6: 구간 음량 복구와 최종 안전 처리

**Files:**
- Create: `engine/src/mrengine/loudness.py`, `engine/tests/test_loudness.py`

**Interfaces:**
- Consumes: `Section` (Task 4)
- Produces:
  - `restore_section_loudness(inst: np.ndarray, mix: np.ndarray, sections: list[Section], max_gain_db: float = 6.0) -> tuple[np.ndarray, list[float]]` — `(보정된 inst, 구간별 적용 게인 dB)`.
  - `finalize(audio: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray` — 피크가 ceiling을 넘으면 전체를 낮춤(정규화. 리미터는 쓰지 않는다).

**알고리즘:** 각 choir 구간에 대해, 구간 바로 앞뒤 각 3초(다른 보컬 구간 제외)의 `rms(inst)/rms(mix)` 비율을 기준 비율로 잡는다. 구간 안의 실제 비율이 기준보다 낮으면 그 차이만큼 게인(`max_gain_db` 상한). 게인은 구간 경계에서 0.2초 동안 선형으로 바꾼다. 앞뒤 기준 구간이 없으면 게인 0dB.

- [ ] **Step 1: 실패하는 테스트**

```python
def test_raises_dropped_section(...):
    # 악기 1kHz 톤, 3~6초만 inst 음량 -6dB, mix는 일정
    out, gains = restore_section_loudness(inst, mix, [Section(3, 6, "choir")])
    assert abs(rms_db(out, 3.3, 5.7) - rms_db(out, 0.5, 2.5)) < 1.0
    assert 5.0 < gains[0] <= 6.0

def test_gain_capped(...):  # -15dB 떨어진 구간 → 게인 정확히 6.0
def test_lead_sections_untouched(...):  # kind="lead"는 게인 0
def test_finalize_peak_never_exceeds_minus_1dbfs(tone):
    out = finalize(tone(440, 1.0, 1.4))
    assert 20*np.log10(np.max(np.abs(out))) <= -1.0 + 1e-3
def test_finalize_leaves_quiet_audio(tone):  # 피크 -6dB면 그대로
```

- [ ] **Step 2: 실패 확인** → FAIL.
- [ ] **Step 3: 구현**.
- [ ] **Step 4: 통과 확인** → 5 passed.
- [ ] **Step 5: 커밋** — `합창 구간 음량 복구와 피크 안전 처리 추가` (왜: 사용자가 미디 프로그램에서 손으로 키우던 작업 자동화, 키운 뒤 클리핑 방지).

---

### Task 7: 잔여 보컬 스캔과 감쇄

**Files:**
- Create: `engine/src/mrengine/residual.py`, `engine/tests/test_residual.py`

**Interfaces:**
- Consumes: `Separator`, `Stems` (Task 3), `Section` (Task 4), `stft`/`istft` (Task 2)
- Produces:
  - `@dataclass ResidualSpan: start_s: float; end_s: float; level_db: float` + `to_dict()`
  - `scan_and_reduce(mr: np.ndarray, separator: Separator, threshold_db: float = -30.0, reduction: float = 0.8) -> tuple[np.ndarray, list[ResidualSpan]]` — MR을 다시 분리해 얻은 보컬 성분의 0.5초 창 RMS가 `MR RMS 대비 threshold_db`를 넘는 구간을 찾고, 그 구간에서만 `mr - reduction * 재분리 보컬`(경계 0.1초 페이드). 전 구간 이하면 입력 그대로 반환.

- [ ] **Step 1: 실패하는 테스트** — `FakeSeparator`(테스트 내부, 미리 정한 보컬 배열을 돌려줌)로.

```python
def test_finds_residual_span(...):
    # MR = 악기 + 2~3초에만 -20dB 노이즈
    out, spans = scan_and_reduce(mr, fake, threshold_db=-30)
    assert len(spans) == 1 and abs(spans[0].start_s - 2) < 0.5

def test_reduces_only_in_span(...):  # 구간 밖 샘플은 입력과 동일
def test_clean_mr_is_unchanged(...):  # 보컬 0 → spans == [] and out == mr
```

- [ ] **Step 2: 실패 확인** → FAIL.
- [ ] **Step 3: 구현**.
- [ ] **Step 4: 통과 확인** → 3 passed.
- [ ] **Step 5: 커밋** — `잔여 보컬 스캔과 감쇄 추가`.

---

### Task 8: 파이프라인, report.json, CLI

**Files:**
- Create: `engine/src/mrengine/pipeline.py`, `engine/src/mrengine/cli.py`, `engine/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1~7의 모든 공개 함수
- Produces:
  - `@dataclass PipelineConfig: strength: float = 0.5; ensemble_method: EnsembleMethod = "avg"; max_gain_db: float = 6.0; residual_threshold_db: float = -30.0; write_mp3: bool = True`
  - `run_pipeline(input_path: Path, out_dir: Path, separators: list[Separator], config: PipelineConfig = PipelineConfig()) -> dict` — 반환값은 report.json에 쓴 dict와 동일.
  - 출력 파일: `out_dir/mr_baseline.wav`, `out_dir/mr_restored.wav` (+ `.mp3` 각각), `out_dir/report.json`
  - `report.json` 키: `input`, `duration_s`, `channels`, `models` (separator name 목록), `config`, `sections` (Section.to_dict 목록), `section_gains_db`, `restore_amount` (구간별 평균 복구량), `residual_spans`, `timings_s` (단계별 초).
  - CLI: `python -m mrengine.cli <input> --out <dir> [--strength 0.5] [--ensemble avg|min_mag|max_mag] [--models a.ckpt,b.yaml]` — 기본 모델은 `DEFAULT_MODELS`.

**흐름:** load → 각 separator.separate → `ensemble`(instrumental들) = baseline, 보컬도 같은 방식으로 앙상블 → `detect_vocal_sections` → `restore_choir` → `restore_section_loudness` → `scan_and_reduce`(첫 번째 separator 사용) → `finalize` (baseline에도 `finalize`를 적용해 A/B 비교 시 음량 조건을 맞춤) → 저장.

- [ ] **Step 1: 실패하는 테스트** — `FakeSeparator`로 GPU 없이.

```python
def test_pipeline_writes_all_outputs(tmp_path, ...):
    rep = run_pipeline(wav, tmp_path / "out", [fake_a, fake_b])
    for f in ["mr_baseline.wav", "mr_restored.wav", "mr_baseline.mp3", "mr_restored.mp3", "report.json"]:
        assert (tmp_path / "out" / f).exists()
    assert json.loads((tmp_path/"out"/"report.json").read_text(encoding="utf-8")) == rep

def test_pipeline_preserves_length_and_channels(...):  # 모노 입력 → 모노 출력, 길이 동일
def test_pipeline_no_vocal_sections_returns_baseline(...):
    # 보컬이 0인 fake → sections == [] and restored ≈ baseline (atol 1e-3)
def test_report_has_required_keys(...):
```

- [ ] **Step 2: 실패 확인** → FAIL.
- [ ] **Step 3: 구현**.
- [ ] **Step 4: 통과 확인** — 전체 `pytest -m "not gpu" -v` 통과.
- [ ] **Step 5: 실제 곡 1개로 수동 확인** — `.venv/Scripts/python -m mrengine.cli "../Test AR/14. 낡은 침대를 타고.mp3" --out out/smoke` → 5개 파일 생성, report.json에 sections가 1개 이상, 처리 시간 기록. 걸린 시간과 GPU 메모리 최대치(`nvidia-smi`)를 `docs/STATE.md`에 적는다.
- [ ] **Step 6: 커밋** — `엔진 파이프라인과 CLI 추가`.

---

### Task 9: Phase 0 일괄 처리, 비교 자료, 청취 판정

**Files:**
- Create: `engine/scripts/phase0_batch.py`
- Modify: `engine/MODELS.md`, `docs/STATE.md`, `todo.md`

**Interfaces:**
- Consumes: `run_pipeline`, `PipelineConfig`, `AudioSeparatorBackend`, `DEFAULT_MODELS`, `load_audio`, `stft`
- Produces: `engine/out/phase0/<곡 이름>/` 아래 파이프라인 출력 + `compare.png` + `engine/out/phase0/listening_sheet.md`

- [ ] **Step 1: 스크립트 작성** — `python scripts/phase0_batch.py --songs "../Test AR" --out out/phase0 [--strengths 0.3,0.5,0.8] [--ensembles avg,min_mag]`
  - 곡마다, 조합마다 `run_pipeline` 실행 (분리 결과는 조합 간에 재사용해서 GPU 시간을 아낀다 — 분리 단계 결과를 `out/phase0/<곡>/stems_cache/`에 저장하고 있으면 읽음).
  - `compare.png`: 가장 강한 choir 구간 ±5초의 스펙트로그램을 위에서부터 원곡 / baseline / restored / (있으면) `Test AR/gaudio/<곡 이름>*` 순으로. choir 구간을 반투명 박스로 표시.
  - `listening_sheet.md`: 곡별로 들어볼 시각(choir 구간 시작 시각), 파일 경로, 그리고 "Gaudio보다 꽉 차는가 / 합창이 새는가 / 이상한 소리가 나는가" 체크 칸.
- [ ] **Step 2: 3곡 실행** — 위 명령 실행. 끝나면 곡별 처리 시간을 `docs/STATE.md`에 기록.
- [ ] **Step 3: 검출 임계값 1차 조정** — 각 곡 report의 choir 구간이 실제 합창 부분(특히 `Do You Hear the People Sing`의 후렴)과 맞는지 `compare.png`로 확인. 어긋나면 Task 4 상수를 조정하고 합성 테스트가 계속 통과하는지 확인 후 커밋.
- [ ] **Step 4: 사용자 청취 판정 (사람이 하는 단계)** — 사용자에게 `listening_sheet.md`와 파일 위치를 전달하고 판정을 받는다. Gaudio 결과물(`Test AR/gaudio/`)이 아직 없으면 이 단계 전에 요청한다.
  - **통과 기준 (스펙 §7): 사용자가 들었을 때 합창 구간 반주가 Gaudio보다 확실히 더 꽉 차게 들린다.**
  - 판정 결과, 선택된 strength·ensemble 조합, 남은 문제를 `docs/STATE.md`와 `MODELS.md`에 기록.
- [ ] **Step 5: 커밋** — `Phase 0 일괄 비교 스크립트와 청취 판정 결과 기록`.
- [ ] **Step 6: 분기**
  - 통과 → `PipelineConfig` 기본값을 선택된 조합으로 바꾸고 커밋. 다음은 웹 MVP 설계(Phase 1) 계획서 작성.
  - 미달 → 기간(Phase 0 시작 후 1주) 안에 원인을 정리해 사용자와 방향을 다시 정한다. 기간을 넘겨 혼자 다듬지 않는다 (반목표 A3).
