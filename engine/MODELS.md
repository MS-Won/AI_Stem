# 분리 모델 기록

모델 목록과 SDR은 설치된 audio-separator 0.48.0의 `--list_models` 기준이다. 모델 파일은 `engine/models/`에 받으며 커밋하지 않는다.
SDR은 목록이 보고하는 값이며 곡 품질 판정이 아니다(Task 9 청취로 판정).

| 파일 | 구조 | 역할 | 목록 SDR | 출처 | 라이선스 | 상업 이용 |
|---|---|---|---|---|---|---|
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | BS-RoFormer (MDXC) | 보컬/반주 (DEFAULT) | vocals 11.8 / inst 16.5 | Viperx BS-RoFormer (audio-separator 모델 목록) | unknown — verify before commercial launch | unknown — verify before commercial launch |
| `htdemucs_ft.yaml` | Demucs v4 (htdemucs_ft) | 4-stem, 반주 = drums+bass+other (DEFAULT) | vocals 10.8 / drums 10.0 / bass 12.0 | Meta `facebookresearch/demucs` | MIT (demucs 저장소 코드 라이선스. 가중치도 같은 저장소 배포이나 출시 전 재확인) | 가능으로 알려짐, 출시 전 재확인 |
| `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | Mel-RoFormer (MDXC) | karaoke 비교 후보 (DEFAULT 아님) | vocals 8.4 / inst 14.7 | aufr33 + Viperx (audio-separator 모델 목록) | unknown — verify before commercial launch | unknown — verify before commercial launch |

## 선정 이유
- BS-RoFormer Viperx 1297: 목록의 RoFormer 중 SDR이 검증되어 있고(inst 16.5) 널리 쓰이는 표준 모델이다. Gabox/unwa 계열은 SDR이 "Unknown"이라 근거가 부족해 제외했다. (inst 16.1의 unwa Inst V2, 16.1의 InstVoc Duality는 후보로 남음.)
- htdemucs_ft: RoFormer와 구조가 달라 앙상블 다양성이 크고, 라이선스가 가장 명확하다.
- karaoke 계열: 합창·백보컬을 보컬로 보지 않을 가능성을 비교하려는 후보. Phase 0 DEFAULT_MODELS에는 넣지 않았다(Task 9에서 필요하면 추가).

## 실행 환경 메모
- 선정한 두 모델 모두 PyTorch(.ckpt/.yaml)라 CUDA(RTX 3060)에서 돈다. ONNX 모델은 쓰지 않았으므로 onnxruntime-gpu 동작은 검증하지 않았다(로그상 CUDAExecutionProvider는 보임).
- 상업 출시 전에 위 "unknown" 항목의 라이선스를 반드시 확인한다. 확인 전까지 해당 모델은 Phase 0 내부 비교용으로만 쓴다.
