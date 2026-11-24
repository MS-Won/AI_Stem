# 할 일 (Phase 0 MR 엔진)

- [x] Task 1: 엔진 뼈대, 작업 문서, 오디오 입출력
- [x] Task 2: STFT 공통 함수와 앙상블 (왕복 복원 atol 1e-4, 프레임 중심 시각 검증, avg/min_mag/max_mag 테스트 8개 통과, 전체 13 passed)
- [x] Task 3: 분리 백엔드 (audio-separator Python API 래퍼, 단위 5개 + gpu 3개 통과. BS-RoFormer 1297 + htdemucs_ft를 실제 30초 구간으로 GPU 실행 확인: 25.0s/1.6GB, 11.6s/1.0GB. 라이선스는 MODELS.md, 일부 unknown)
- [x] Task 4: 보컬·합창 구간 검출 (합성곡에서 choir 3~6초·lead 6~9초 검출, 무음/0.3초 블립 제외, frames_mask 포함 테스트 7개 통과. 임계값은 sections.py 상수, Task 9에서 실제 곡 조정. 리뷰 반영(R7): 합창 위주 곡 대비 절대 폭 3kHz + 20% 백분위 기준, 활성 프레임만 choir 판정, 테스트 10개)
- [x] Task 5: 합창 구간 반주 복구 (합성 구멍 실험에서 1kHz 대역 복구 strength=1.0 → 원래의 99.5%(구멍 10%), 0.5 → 54.9%. 구간 밖 샘플 불변, I' 칸별 |I'|<=|M| 정확 보장(restore_choir_bins), 재합성 후 초과는 조밀 노이즈 픽스처에서 악화 없음·두 픽스처 모두 R9 초과 에너지 기준 통과, 270초 스테레오 7~8초/피크 1.7GB, 전체 36 passed)
- [x] Task 6: 구간 음량 복구와 최종 안전 처리 (합성 -6dB 하락 구간에서 게인 5~6dB·구간/바깥 음량차 1dB 미만, -15dB 하락은 게인 정확히 6.0 상한, lead 구간 불변, 올리기만 함, 기준 구간 0.5초 미만이면 0dB, finalize 피크 -1dBFS 이하·작은 신호 불변·무음 처리, 전체 44 passed)
- [ ] Task 7: 잔여 보컬 스캔과 감쇄
- [ ] Task 8: 파이프라인, report.json, CLI
- [ ] Task 9: Phase 0 일괄 처리, 비교 자료, 청취 판정
