# 현재 상태

- 브랜치: `phase0-engine`
- 단계: Phase 0 (MR 엔진), **Task 5 완료** (`restore.py`: `restore_choir`, choir 구간만 믹스-보컬 나머지로 복구, 경계 0.1초 램프, 칸별 |I'|<=|M| 클램프 + 재투영 재클램프), Task 4 완료 (구간 검출 `sections.py`: `detect_vocal_sections`, `frames_mask`; 합창 판정은 절대 폭 3kHz OR 하위 20% 백분위 기준 1.5배 OR 평탄도) — Task 3까지: STFT/앙상블(Task 2)에 이어 분리 백엔드(`separate.py`: `AudioSeparatorBackend`, `normalize_stems`, `DEFAULT_MODELS` 2개, `MODEL_DIR`)와 `engine/MODELS.md`(라이선스 일부 unknown, 상업 출시 전 확인)

## 다음 할 일
1. (완료) Task 1, 2, 3
2. (완료) Task 4
3. (완료) Task 5
4. Task 6: 구간 음량 복구와 최종 안전 처리

전체 목록은 `todo.md`, 계획은 `docs/superpowers/plans/2026-11-24-phase0-mr-engine.md`.
