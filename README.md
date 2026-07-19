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

| 구역 | Stage | 상태 |
|---|---|---|
| Data foundation | 2 Demo Loader, 4 Feature Engine | 구현 + 단위 테스트 통과 |
| | 3 State Replay | 인터페이스만 (robosuite 구현 대기) |
| Automatic segmentation | 5–7 | 미착수 |
| FCM / Degradation | 8–12 | 미착수 |

Stage 13 이후(reward/policy)는 설계상 provisional이며 아직 구현 대상이 아니다.

## 구조

```
src/bfd_pipeline/
├── core/types.py           # 공통 dataclass (EpisodeData, RawTrajectory, FeatureTrajectory)
├── data/demo_io.py         # Stage 2: HDF5 -> EpisodeData
├── envs/base_adapter.py    # Stage 3: replay 인터페이스 + synthetic 생성기
└── features/               # Stage 4: feature 함수 / registry / engine / robust scaler
```
