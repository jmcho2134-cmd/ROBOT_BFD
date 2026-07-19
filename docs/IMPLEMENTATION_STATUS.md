# Implementation Status

Status vocabulary (per blueprint §24):
`NOT_STARTED` · `INTERFACE_DEFINED` · `IMPLEMENTED_UNVERIFIED` · `GATE_FAILED` · `GATE_PASSED`

`설계 완료`와 `코드 완료`는 다른 상태다. 아래는 코드 기준이다.

## 핵심 사실

전체 파이프라인(Stage 1–12)이 **경량 합성 Pick-and-Place 시뮬레이터** 위에서
end-to-end로 실행되며, `Original > Mild > Medium > Severe` degradation family를
실제 시뮬레이터 rollout으로 검증한다. robosuite/MuJoCo 및 실제 demo는 아직 없다.
합성 시뮬레이터는 실데이터를 붙이기 전 Stage 5–12를 실행·검증하기 위한 것이며,
`BaseReplayAdapter` 계약을 그대로 구현하므로 robosuite adapter로 교체 시
Stage 5–12 코드는 바뀌지 않는다.

| Stage | Module | Status | Gate | Notes |
|---|---|---|---|---|
| 1 Demo Collection | `envs/synthetic_env.py` (scripted expert) | IMPLEMENTED_UNVERIFIED | — | 합성 데모 생성 (성공률 ~0.95). 실 teleop 데이터 아님 |
| 2 Demo Loader | `data/demo_io.py` | GATE_PASSED* | unit PASS | HDF5 자동감지 + NaN/shape reject |
| 3 State Replay | `envs/base_adapter.py`, `synthetic_env.py` | GATE_PASSED* (synthetic) | restore err=0 | 합성 env는 exact restore. robosuite adapter 미구현 |
| 4 Feature Engine | `features/*` | GATE_PASSED* | unit PASS | 10 feature, Gate B 강제, train-only scaler |
| 5 Feature Profiling | `features/profiler.py` | IMPLEMENTED_UNVERIFIED | — | name-free 점수 + bootstrap. 거리·contact structural, jerk 비-structural로 분리됨 |
| 6 Phase Segmentation | `segmentation/{evidence,segmenter}.py` | IMPLEMENTED_UNVERIFIED | — | DP + 후보풀 + canonical alignment. 합성 데모에서 K=5, conf 1.0 |
| 7 Subgoal Inference | `segmentation/subgoal_inference.py` | IMPLEMENTED_UNVERIFIED | — | Change/Hold/Passive + completion/degradation score |
| 8 Perturbation Rollout | `consequence/rollout.py` | GATE_PASSED* | Gate8 \|resid\|@0≈1e-7 | 동일 state branch rollout |
| 9 Residual FCM | `consequence/fcm.py` | IMPLEMENTED_UNVERIFIED | — | sklearn MLP ensemble. R2(med)≈0.7 (held-out row split) |
| 10 Hypothesis | `degradation/hypothesis.py` | IMPLEMENTED_UNVERIFIED | — | change_opposition / hold_violation |
| 11 Candidate + Screening | `degradation/candidate.py` | IMPLEMENTED_UNVERIFIED | — | subspace×direction×start/dur, FCM Top-K + diversity |
| 12 Simulator Validation | `degradation/validator.py` | IMPLEMENTED_UNVERIFIED | — | λ bracketing+binary search, nested levels, monotonic family |
| 13–16 | `provisional_bfd/` | NOT_STARTED | — | 설계상 provisional |

\* 합성 시뮬레이터 기준 통과. robosuite/실데이터에서 재검증 필요.

## 검증된 것 (pytest 20 passed)
- Stage 2/4 단위 테스트, Stage 5–7 (structural 분리, 최소 phase 길이, canonical K 안정, subgoal 존재, degradation 부호).
- 통합 테스트: `run_pipeline`이 Gate8<1e-3, canonical K 3–8, FCM screening이 candidate 축소, 그리고 **monotonic·success-preserving degradation family ≥1** 생성.
- 드라이버: `python -m bfd_pipeline.pipeline` → 24 demo에서 validated family ~12개.

## 알려진 단순화 (실데이터 전 개선 대상)
1. **합성 env 한계**: rotation action(3:6)은 dynamics에 영향 없음(inert). 접촉/파지가 단순화됨.
2. **FCM screening**: 소수 demo context 평균. NDCG/Top-K recall vs random baseline(Gate 9)은 아직 정식 측정 안 함.
3. **Stage 6**: leave-one-feature-out / temporal jitter 안정성 리포트(§11.11)는 미구현(canonical confidence만 사용).
4. **reliability weight**: 현재 `structural×confidence`. blueprint 리뷰 A의 strength/reliability 분리는 유보.

## 다음 단계
1. robosuite adapter(Stage 3) 구현 + 실제 demo → 합성 결과 재현.
2. Gate 9(FCM Top-K recall vs random) 정식 측정 추가.
3. Stage 6 안정성 리포트(§11.11 B/C) 구현.
