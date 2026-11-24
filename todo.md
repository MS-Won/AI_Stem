# 할 일 (Phase 0 MR 엔진)

- [x] Task 1: 엔진 뼈대, 작업 문서, 오디오 입출력
- [x] Task 2: STFT 공통 함수와 앙상블 (왕복 복원 atol 1e-4, 프레임 중심 시각 검증, avg/min_mag/max_mag 테스트 8개 통과, 전체 13 passed)
- [x] Task 3: 분리 백엔드 (audio-separator Python API 래퍼, 단위 5개 + gpu 3개 통과. BS-RoFormer 1297 + htdemucs_ft를 실제 30초 구간으로 GPU 실행 확인: 25.0s/1.6GB, 11.6s/1.0GB. 라이선스는 MODELS.md, 일부 unknown)
- [x] Task 4: 보컬·합창 구간 검출 (합성곡에서 choir 3~6초·lead 6~9초 검출, 무음/0.3초 블립 제외, frames_mask 포함 테스트 7개 통과. 임계값은 sections.py 상수, Task 9에서 실제 곡 조정. 리뷰 반영(R7): 합창 위주 곡 대비 절대 폭 3kHz + 20% 백분위 기준, 활성 프레임만 choir 판정, 테스트 10개)
- [x] Task 5: 합창 구간 반주 복구 (합성 구멍 실험에서 1kHz 대역 복구 strength=1.0 → 원래의 99.5%(구멍 10%), 0.5 → 54.9%. 구간 밖 샘플 불변, I' 칸별 |I'|<=|M| 정확 보장(restore_choir_bins), 재합성 후 초과는 조밀 노이즈 픽스처에서 악화 없음·두 픽스처 모두 R9 초과 에너지 기준 통과, 270초 스테레오 7~8초/피크 1.7GB, 전체 36 passed)
- [x] Task 6: 구간 음량 복구와 최종 안전 처리 (합성 -6dB 하락 구간에서 게인 5~6dB·구간/바깥 음량차 1dB 미만, -15dB 하락은 게인 정확히 6.0 상한, lead 구간 불변, 올리기만 함, 기준 구간 0.5초 미만이면 0dB, finalize 피크 -1dBFS 이하·작은 신호 불변·무음 처리, 전체 44 passed)
- [x] Task 7: 잔여 보컬 스캔과 감쇄 (2~3초에만 보컬이 섞인 합성 MR에서 구간 1개(2~3초) 검출, 분리기 호출 1회, 구간 밖 샘플 비트 동일, 보컬 0이면 입력 그대로·spans 빈 목록, 무음 MR 처리, FakeSeparator는 conftest에 추가, 전체 48 passed)
- [x] Task 8: 파이프라인, report.json, CLI (FakeSeparator 테스트 7개 통과, 전체 55 passed. separate_all/stems 재사용(R4)으로 분리 1회만 호출 확인. 실제 곡 14번(274초 스테레오) 스모크: 5개 파일 생성, 구간 53개(합창 29), 총 658초, GPU 최대 4355MiB(기준 1117 포함). 합창 구간 평균 복구량이 0.000~0.014로 매우 작음, 잔여 스캔이 270초로 가장 느림 -> Task 9에서 조정)
- [ ] Task 9: Phase 0 일괄 처리, 비교 자료, 청취 판정
  - [x] Step 1: `engine/scripts/phase0_batch.py` (분리 캐시 stems_cache/*.npy, 탐색 조합은 잔여 스캔 생략(`PipelineConfig.residual=False`), 최종 조합만 잔여 스캔, compare.png, 한국어 listening_sheet.md, `--reuse`) + 진단 스크립트 `phase0_diagnose.py`
  - [x] Step 2: 3곡 실행 (곡별 시간은 docs/STATE.md). 최종 리뷰 수정 후 3곡 모두 곡별 실행으로 재생성 완료
  - [x] Step 3: 검출 임계값 1차 조정 (V_RATIO 0.25→0.12, 활성 평활 0.5→1.5초, choir 평활 4초 분리, 폭 배수 1.5→1.4. 합창 구간 수 14/29/53 → 16/22/13, 조각 구간 감소). 전체 58 passed
  - [ ] Step 4: 사용자 청취 판정 (Gaudio 파일 필요, 컨트롤러가 진행)
  - [x] Step 5: 커밋
  - [ ] Step 6: 판정에 따른 분기

## Phase 1 진입 조건 (최종 리뷰 결과, 2026-11-24)
- [ ] BS-RoFormer(`DEFAULT_MODELS[0]`) 라이선스 확인 또는 대체 모델 선택 (MODELS.md에 unknown 있음)
- [ ] 작업별(per-job) work_dir: 지금은 `input.wav` 이름이 고정이라 동시 작업 시 충돌
- [ ] ffmpeg 실패 시 stderr를 오류 메시지에 포함
- [ ] sections/envelope 메모리 (긴 곡·동시 작업 시 RAM 2~3GB/곡)
- [ ] 잔여 스캔 비용 (전체 시간의 약 45%) 줄이기 또는 선택 사항으로
- [ ] 음량 목표 결정: "원곡 음량 흐름 따르기"(현재, 원곡보다 커질 수 있음) vs "잃어버린 반주만 복구"
- 참고: Review Focus 2에서 계획한 테스트 이름은 `test_normalize_stems_restores_mono_from_stereo` + gpu 유니코드 테스트가 대신 다룬다.
