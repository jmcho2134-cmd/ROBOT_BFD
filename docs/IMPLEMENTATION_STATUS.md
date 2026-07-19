# Implementation Status

Status vocabulary (per blueprint §24):
`NOT_STARTED` · `INTERFACE_DEFINED` · `IMPLEMENTED_UNVERIFIED` · `GATE_FAILED` · `GATE_PASSED`

`설계 완료`와 `코드 완료`는 다른 상태다. 아래는 코드 기준이다.

| Stage | Module | Status | Gate | Notes |
|---|---|---|---|---|
| 1 Demo Collection | (data provisioning) | NOT_STARTED | — | 실제 demo.hdf5 미확보 |
| 2 Demo Loader | `data/demo_io.py` | IMPLEMENTED_UNVERIFIED | unit PASS | robomimic/flat HDF5 자동 감지, NaN/shape 검증, 단위 테스트 통과. 실데이터 검증 전 |
| 3 State Replay Adapter | `envs/base_adapter.py` | INTERFACE_DEFINED | — | 추상 인터페이스 + synthetic 생성기만. robosuite/MuJoCo 구현은 `sim` extra + 실데이터 필요 |
| 4 Feature Engine | `features/{feature_functions,registry,engine,normalization}.py` | IMPLEMENTED_UNVERIFIED | unit PASS | 10개 feature, Gate B(name-only) 강제, robust scaler(train-only fit), 단위 테스트 통과 |
| 5 Feature Profiling | `features/profiler.py` | NOT_STARTED | — | |
| 6 Phase Segmentation | `segmentation/` | NOT_STARTED | — | 최고 우선순위. 착수 전 blueprint §11.4 reliability weight 정의 확정 필요 |
| 7 Subgoal Inference | `segmentation/subgoal_inference.py` | NOT_STARTED | — | |
| 8–12 | `consequence/`, `degradation/` | NOT_STARTED | — | |

## 현재 검증된 것
- `pytest` 14개 통과 (demo loader 4, feature engine 6, registry 4).
- synthetic Pick-and-Place trajectory 위에서 feature 계산이 물리적으로 그럴듯한 값을 반환 (approach 시 eef_object_dist 감소, contact binary, object_goal_dist 수렴).

## 다음 단계
1. 실제 `reference_data/demos/*.hdf5` 확보 → Stage 2를 실데이터로 재검증.
2. robosuite 설치(`pip install -e .[sim]`) 후 `robosuite_adapter.py` 구현 → Gate 3(state restore) 검증.
3. Stage 5 profiler 착수. 단, 그 전에 §11.4 가중치 정의(신뢰성 전용 여부)를 문서에서 확정.
