# CLAUDE.md

## 세션 시작 시
**먼저 `docs/STATE.md`를 읽어라.** 현재 진행 상황과 다음 할 일이 거기 있다.

## 프로젝트
합창이 있는 곡에서 합창 구간까지 꽉 찬 MR(반주)을 만드는 Python 엔진 `mrengine` (`engine/`).

## 문서 지도
- 설계: `docs/superpowers/specs/2026-11-24-choir-mr-mvp-design.md`
- 계획: `docs/superpowers/plans/2026-11-24-phase0-mr-engine.md`
- 현재 상태: `docs/STATE.md`
- 할 일: `todo.md`

## 환경
- Python 3.12, 가상환경 `engine/.venv` (git 제외). ffmpeg/ffprobe는 PATH에 있어야 한다.
- 테스트: `engine/`에서 `.venv/Scripts/python -m pytest` (GPU 없이: `-m "not gpu"`)
- 설치: `pip install -e ".[gpu]" pytest`
- PyTorch CUDA 확인: `.venv/Scripts/python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
  기대 출력: `True NVIDIA GeForce RTX 3060`
- **주의**: `pip install -e ".[gpu]"`는 CPU 전용 torch를 깐다. 설치 직후 CUDA 휠로 교체해야 한다:
  `.venv/Scripts/python -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu130`
  (검증 버전: torch 2.14.1+cu130, 드라이버 610.88, audio-separator 0.48.0)

## 함정 (반드시 지킬 것)
- 오디오 배열은 `np.ndarray` float32 `(channels, samples)`, 44100Hz. 모노 입력을 받고 출력 채널 수는 입력을 따른다.
- 출력 길이는 입력 길이와 정확히 같다 (길면 자르고 짧으면 0으로 채움).
- **반목표 A1**: 복구는 원곡 믹스에 있는 정보만 쓴다. 생성 모델로 채우지 않는다. 어떤 시간-주파수 칸에서도 MR 크기가 원곡 믹스 크기를 넘지 않는다.
- 출력은 WAV 16-bit 44.1kHz + MP3 320kbps, 최종 피크 -1 dBFS 이하.
- `Test AR/` 음원, `engine/out/`, `engine/models/`는 절대 커밋하지 않는다.
- 경로에 한글과 `#`가 있다 (예: `#13 낡은 침대.mp3`). 외부 도구에는 ASCII 임시 이름을 쓴다.
- 외부 라이브러리 API는 계획서 예시를 믿지 말고 설치된 버전의 실제 시그니처를 확인한다.

## 작업 규칙
- TDD: 실패하는 테스트 먼저. pytest가 통과한 뒤에만 커밋한다.
- 커밋 메시지는 한국어로, 왜 그렇게 했는지를 남긴다.
- 작업이 끝나면 `docs/STATE.md`와 `todo.md`를 갱신한다.
