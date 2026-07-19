# ROBOT_BFD

Feature-Function-Only Phase-Conditioned Structured Degradation pipeline for
learning from suboptimal robot demonstrations (robosuite + MuJoCo).

전체 설계는 [`BfD_code_pipeline_overview.md`](BfD_code_pipeline_overview.md),
구현 진행 상태는 [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md)
참고.

## 설치

```bash
pip install -e .            # core (numpy/scipy/h5py) + Stage 1-7 pure-python
pip install -e ".[dev]"     # + pytest
pip install -e ".[sim]"     # + robosuite/mujoco (Stage 3 replay, Stage 8+ rollouts)
```

## 테스트

```bash
pytest -q
```

## 현재 구현 범위

**Stage 1–12 전체가 합성 시뮬레이터 위에서 end-to-end로 실행**되며
`Original > Mild > Medium > Severe` degradation family를 시뮬레이터로 검증한다.

```bash
python -m bfd_pipeline.pipeline    # 데모 -> ... -> validated degradation families
```

| 구역 | Stage | 상태 |
|---|---|---|
| Data foundation | 1–4 | 구현 + 테스트 통과 (Stage 3는 합성 env; robosuite 대기) |
| Automatic segmentation | 5–7 | 구현 (합성 데모에서 canonical K=5, conf 1.0) |
| FCM | 8–9 | 구현 (Gate8 통과, R2≈0.7) |
| Degradation | 10–12 | 구현 (monotonic·success-preserving family 생성) |

세부 상태·단순화 목록은 [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md).
Stage 13 이후(reward/policy)는 설계상 provisional이며 아직 구현 대상이 아니다.

## 구조

```
src/bfd_pipeline/
├── core/types.py           # 공통 dataclass (모든 stage 계약)
├── data/demo_io.py         # Stage 2: HDF5 -> EpisodeData
├── envs/                   # Stage 3: replay 인터페이스 + 합성 Pick-and-Place 시뮬레이터
├── features/               # Stage 4-5: feature 함수/registry/engine/scaler + profiler
├── segmentation/           # Stage 6-7: evidence/segmenter/subgoal_inference
├── consequence/            # Stage 8-9: perturbation rollout + residual FCM
├── degradation/            # Stage 10-12: hypothesis/candidate/validator
└── pipeline.py             # Stage 1-12 end-to-end 드라이버
```

## 참고

합성 시뮬레이터는 robosuite/MuJoCo와 실제 demo가 없이 Stage 5–12를 실행·검증하기
위한 것이다. `BaseReplayAdapter` 계약을 그대로 구현하므로, robosuite adapter로
교체해도 Stage 5–12 코드는 바뀌지 않는다.
