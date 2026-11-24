# 현재 상태

- 브랜치: `phase0-engine`
- 단계: Phase 0 (MR 엔진), **Task 7 완료** (`residual.py`: `scan_and_reduce`가 MR을 재분리해 0.5초 창 RMS가 MR RMS 대비 -30dB를 넘는 구간만 `mr - 0.8*보컬`로 감쇄(0.1초 페이드), `ResidualSpan` 반환), Task 6 완료 (`loudness.py`: `restore_section_loudness` 앞뒤 3초 기준 비율로 choir 구간 게인(0~6dB, 0.2초 램프), `finalize` 피크 -1dBFS 정규화), Task 5 완료 (`restore.py`: `restore_choir`, choir 구간만 믹스-보컬 나머지로 복구, 경계 0.1초 램프, 칸별 |I'|<=|M| 클램프(`restore_choir_bins`), 채널별 처리로 메모리 절감), Task 4 완료 (구간 검출 `sections.py`: `detect_vocal_sections`, `frames_mask`; 합창 판정은 절대 폭 3kHz OR 하위 20% 백분위 기준 1.5배 OR 평탄도) — Task 3까지: STFT/앙상블(Task 2)에 이어 분리 백엔드(`separate.py`: `AudioSeparatorBackend`, `normalize_stems`, `DEFAULT_MODELS` 2개, `MODEL_DIR`)와 `engine/MODELS.md`(라이선스 일부 unknown, 상업 출시 전 확인)

## 다음 할 일
1. (완료) Task 1, 2, 3
2. (완료) Task 4
3. (완료) Task 5
4. (완료) Task 6
5. (완료) Task 7
6. Task 8: 파이프라인, report.json, CLI

전체 목록은 `todo.md`, 계획은 `docs/superpowers/plans/2026-11-24-phase0-mr-engine.md`.
