# 할 일 (Phase 0 MR 엔진)

- [x] Task 1: 엔진 뼈대, 작업 문서, 오디오 입출력
- [x] Task 2: STFT 공통 함수와 앙상블 (왕복 복원 atol 1e-4, 프레임 중심 시각 검증, avg/min_mag/max_mag 테스트 8개 통과, 전체 13 passed)
- [x] Task 3: 분리 백엔드 (audio-separator Python API 래퍼, 단위 5개 + gpu 3개 통과. BS-RoFormer 1297 + htdemucs_ft를 실제 30초 구간으로 GPU 실행 확인: 25.0s/1.6GB, 11.6s/1.0GB. 라이선스는 MODELS.md, 일부 unknown)
- [ ] Task 4: 보컬·합창 구간 검출
- [ ] Task 5: 합창 구간 반주 복구 (핵심 차별점)
- [ ] Task 6: 구간 음량 복구와 최종 안전 처리
- [ ] Task 7: 잔여 보컬 스캔과 감쇄
- [ ] Task 8: 파이프라인, report.json, CLI
- [ ] Task 9: Phase 0 일괄 처리, 비교 자료, 청취 판정
