# 현재 상태

- 브랜치: `phase0-engine`
- 단계: Phase 0 (MR 엔진), **Task 9 Step 1~3·5 완료, Step 4(사용자 청취 판정) 대기** (아래 'Task 9 결과'). Task 8 완료 (`pipeline.py`: `run_pipeline`/`separate_all`/`PipelineConfig`, `stems` 인자로 분리 결과 재사용, `report.json`; `cli.py`; 스모크: 14번 곡 274초 스테레오, 총 658초(분리 360, 잔여 스캔 270, 나머지 약 28), GPU 최대 4355MiB(기준 1117 포함), 구간 53개/합창 29개, 합창 구간 평균 복구량 0.000~0.014로 작음, 잔여 구간 14개), Task 7 완료 (`residual.py`: `scan_and_reduce`가 MR을 재분리해 0.5초 창 RMS가 MR RMS 대비 -30dB를 넘는 구간만 `mr - 0.8*보컬`로 감쇄(0.1초 페이드), `ResidualSpan` 반환), Task 6 완료 (`loudness.py`: `restore_section_loudness` 앞뒤 3초 기준 비율로 choir 구간 게인(0~6dB, 0.2초 램프), `finalize` 피크 -1dBFS 정규화), Task 5 완료 (`restore.py`: `restore_choir`, choir 구간만 믹스-보컬 나머지로 복구, 경계 0.1초 램프, 칸별 |I'|<=|M| 클램프(`restore_choir_bins`), 채널별 처리로 메모리 절감), Task 4 완료 (구간 검출 `sections.py`: `detect_vocal_sections`, `frames_mask`; 합창 판정은 절대 폭 3kHz OR 하위 20% 백분위 기준 1.5배 OR 평탄도) — Task 3까지: STFT/앙상블(Task 2)에 이어 분리 백엔드(`separate.py`: `AudioSeparatorBackend`, `normalize_stems`, `DEFAULT_MODELS` 2개, `MODEL_DIR`)와 `engine/MODELS.md`(라이선스 일부 unknown, 상업 출시 전 확인)

## 다음 할 일
1. (완료) Task 1, 2, 3
2. (완료) Task 4
3. (완료) Task 5
4. (완료) Task 6
5. (완료) Task 7
6. (완료) Task 8
7. Task 9: Step 1~3, 5 완료. 남은 것: (a) 2·3번 곡 출력 다시 만들기 `cd engine && .venv/Scripts/python scripts/phase0_batch.py --songs "../Test AR" --out out/phase0 --reuse` (분리 캐시 있음, 약 13분), (b) Gaudio 결과를 `Test AR/gaudio/`에 넣고 같은 명령 재실행, (c) 사용자 청취 판정(Step 4) → 분기(Step 6)

전체 목록은 `todo.md`, 계획은 `docs/superpowers/plans/2026-11-24-phase0-mr-engine.md`.

## Task 9 결과 (2026-11-24)

### 왜 복구량이 작은가 (진단, `engine/scripts/phase0_diagnose.py`)
- 쉽게 말해: **지금 모델 조합에서는 '빈 곳 채우기'가 채울 재료를 거의 갖고 있지 않다.**
- BS-RoFormer의 반주 출력은 3곡 중 2곡(#13, 낡은 침대)에서 사실상 `원곡 - 보컬`이다 (원곡-보컬-반주 잔차가 반주 대비 -70.8 / -70.5dB). 유봉은 -12.4dB로 그 정도는 아니지만, 합창 구간 |R| 대비 반주 차이는 중앙값 -0.78dB로 역시 작다. htdemucs_ft는 -12 ~ -20dB.
- 복구 재료 R = 원곡 - 보컬 추정치이므로 |R| ≈ |반주|. 합창 구간에서 반주가 |R|보다 작은 정도는 중앙값 -0.05 ~ -0.8dB뿐이다.
- 목표치 target = min(ref·|원곡|, |R|)에서 |R|이 묶는 칸이 53~79% (곡별 가중 평균). ref·|원곡|은 원곡에 합창 자체가 들어 있어 클 수밖에 없고, |R| 상한이 합창 누출(A2)을 막는 역할이라 풀 수 없다.
- strength=1.0에서도 합창 구간 반주 에너지 증가(구간 중앙값): 곡별 0.18 / 0.01 / 0.44 dB (avg). 보컬을 min_mag로 잡으면 0.44 / 0.22 / 0.85 dB. 예외: #13의 01:23(2.7초, 거의 무반주)에서 16.5dB — 모델이 반주를 거의 0으로 만든 곳을 R로 채우는 경우라 합창 누출 위험 지점.
- 결론: 파라미터(strength, ref 통계, 앙상블)로는 의미 있는 채움을 만들 수 없다. 알고리즘은 바꾸지 않았다. baseline과 restored의 들리는 차이는 대부분 Task 6 음량 보정(0~6dB)과 잔여 보컬 제거에서 온다. 합창 구간 반주가 정말 '구멍'인지(모델이 반주를 보컬 쪽으로 가져갔는지)는 원곡에 합창이 섞여 있어 수치만으로는 판별할 수 없다 → 청취로 판정.
- 더 채우려면 보컬 추정치 V 쪽에서 재료를 가져와야 하는데(합창이 같이 들어옴, A2), 이것은 설계 변경이라 사용자와 방향을 정할 일이다.

### 바꾼 것
- `PipelineConfig.residual`(기본 True): False면 잔여 스캔 생략 (탐색 조합용).
- `pipeline.finalize_pair`: baseline과 restored를 같은 배율로 피크 처리. 따로 정규화하면 음량 보정으로 피크가 커진 restored가 통째로 최대 6dB 작아져 같은 시각 A/B 비교가 불공정했다 (1차 실행에서 3번 곡 restored-baseline -5.6/-6.0dB 관측).
- 구간 검출 상수 (실제 곡에서 1초짜리 choir/lead 조각이 번갈아 나오고, 큰 합창+관현악에서 보컬 활성이 끊기던 문제): V_RATIO_THRESHOLD 0.25→0.12, MEDIAN_FILTER_S 0.5→1.5, 새 CHOIR_MEDIAN_FILTER_S 4.0(choir/lead 판정 평활), CHOIR_WIDTH_RATIO 1.5→1.4. 2초보다 짧은 합창 덩어리는 이제 lead로 묶인다.
- 구간 수 (전체/합창/합창 초): #13 26/14/20초 → 38/16/83초, 낡은 침대 52/29/50초 → 48/22/129초, 유봉 68/53/110초 → 24/13/221초.
- 한계: 프레임 대역 폭 특징이 실제 곡에서 합창과 독창을 잘 가르지 못한다 (#13 독창·합창 폭이 모두 600~800Hz). #13 후반(02:22~03:15) 합창은 일부만 choir. 유봉은 거의 전곡이 choir. 정답 없이 조정했으므로 청취표의 '실제로 합창인가' 칸으로 확인받는다.

### 곡별 처리 시간 (RTX 3060, 분리 캐시 생성 + 8조합 탐색 + 최종 1조합)
| 곡 | 길이 | 분리(2모델) | 탐색 8조합(잔여 스캔 없음) | 최종(잔여 스캔 포함) | 합계 |
|---|---|---|---|---|---|
| #13 Do You Hear the People Sing (모노) | 3:20 | 259초 | 66~69초 | 212~220초(잔여 194) | 약 9.5분 |
| 14. 낡은 침대를 타고 | 4:34 | 354초 | 168초 | 295초(잔여 265) | 약 13.6분 |
| 14. 소리도둑 유봉 AR | 4:29 | 348초 | 171초 | 290초(잔여 259) | 약 13.5분 |
- GPU 최대 4428MiB(기준 약 900~1100 포함). 모노 곡도 문제없이 분리됨. 한 곡 실서비스 기준(분리+최종) 약 8~11분, 잔여 스캔이 절반 가까이 차지.
- 2차 실행(`finalize_pair` 반영)은 #13까지 끝나고 시스템 메모리 부족으로 중단됨 → 2·3번 곡은 위 '다음 할 일' 명령으로 재생성 필요. 1차 실행 숫자는 `out/phase0/<곡>/summary_run1.json`.

### 추천 조합
- 세 곡 모두 `s1.0_avg` (잔여 스캔 포함 최종본은 `out/phase0/<곡>/final_s1.0_avg/`). 스펙트럼 복구가 작아서 strength는 최대로 두고, 반주가 가장 두꺼운 avg를 기본으로 했다. min_mag는 복구 여지가 약 2배지만 반주 자체가 얇아질 수 있어 비교용.
- 청취표: `engine/out/phase0/listening_sheet.md` (재생성 후 사용).
