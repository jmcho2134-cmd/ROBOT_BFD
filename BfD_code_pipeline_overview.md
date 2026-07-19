# BfD 코드 파이프라인 설계 문서
## Feature-Function-Only 기반 Phase-Conditioned Structured Degradation Reward Learning

---

## 1. 프로젝트 목표

본 프로젝트의 목표는 **성공했지만 비효율적인 suboptimal demonstration**으로부터 단순 모방이 아니라, 시연자보다 더 효율적인 정책을 학습하는 것이다.

핵심 아이디어는 다음과 같다.

1. 사람이 suboptimal demonstration을 수집한다.
2. 사람이 필요한 feature의 **계산 함수만 정의한다**.
3. 알고리즘이 feature trajectory를 분석해 phase와 subgoal을 자동 추론한다.
4. Forward Consequence Model(FCM)이 action perturbation이 feature에 미치는 결과를 예측한다.
5. phase subgoal을 열화시키는 action 방향을 탐색한다.
6. simulator가 실제 degradation의 성공 여부와 단조성을 검증한다.
7. `Original > Mild > Medium > Severe` preference hierarchy를 만든다.
8. reward network가 전체 feature와 preference를 이용해 보상 함수를 학습한다.
9. 학습된 reward로 policy를 보수적으로 개선한다.

---

## 2. 가장 중요한 설계 원칙

### 2.1 사람이 정의하는 것은 feature 계산 함수뿐이다

연구자가 작성하는 정보는 다음 두 가지다.

```python
FEATURE_FUNCTIONS = {
    "eef_object_dist": compute_eef_object_dist,
    "object_goal_dist": compute_object_goal_dist,
    "contact": compute_contact,
    "joint_energy": compute_joint_energy,
}
```

사람이 직접 지정하지 않는 정보:

```text
progress / event / quality
boundary=True / False
higher_is_better / higher_is_worse
phase 이름
phase 개수
subgoal
degradation 방향
action perturbation 방향
```

### 2.2 Quality feature의 방향은 중간 단계에서 정하지 않는다

예를 들어 다음 feature가 있다고 하자.

```text
joint energy
jerk
path length
action magnitude
object slip
```

중간 모듈이 다음과 같은 방향을 하드코딩하지 않는다.

```text
energy가 크면 나쁘다
jerk가 크면 나쁘다
slip이 작으면 좋다
```

대신 모든 degradation rollout에서 이 feature 값을 저장하고, preference reward learning 단계에서 방향과 중요도를 학습한다.

### 2.3 FCM은 최종 판정기가 아니라 후보 선별기다

FCM은 simulator 실행 비용을 줄이기 위한 screening model이다.

```text
많은 action perturbation 후보
        ↓
FCM 예측
        ↓
Top-K 후보 선별
        ↓
실제 simulator rollout
        ↓
최종 degradation 검증
```

FCM이 예측한 결과만으로 preference를 만들지 않는다.

### 2.4 모든 Stage는 독립적으로 검증한다

각 Stage는 반드시 다음 네 가지를 제공한다.

```text
코드
테스트
시각화 또는 로그
저장 artifact
```

이전 Stage가 통과하지 않으면 다음 Stage로 넘어가지 않는다.

---

# 3. 전체 데이터 흐름

```text
[1] Teleoperation Demonstration
        ↓
demo.hdf5
        ↓
[2] Demo Loader
        ↓
EpisodeData
        ↓
[3] Robosuite Adapter / Replay
        ↓
RawTrajectory
        ↓
[4] Feature Engine
        ↓
FeatureTrajectory
        ↓
[5] Automatic Feature Profiler
        ↓
FeatureProfile
        ↓
[6] Multivariate Phase Segmentation
        ↓
PhaseSegmentation
        ↓
[7] Phase Subgoal Inference
        ↓
PhaseSubgoal
        ↓
[8] Perturbation Rollout Collection
        ↓
FCM Dataset
        ↓
[9] Residual FCM Training
        ↓
Residual FCM
        ↓
[10] Degradation Hypothesis Generation
        ↓
DegradationHypothesis
        ↓
[11] Candidate Search
        ↓
ScreenedCandidate
        ↓
[12] Exact Simulator Validation
        ↓
ValidatedFamily
        ↓
[13] Preference Dataset
        ↓
PreferencePair
        ↓
[14] Reward Learning
        ↓
Reward Model
        ↓
[15] Conservative Policy Improvement
        ↓
Improved Policy
        ↓
[16] End-to-End Evaluation
```

---

# 4. 권장 프로젝트 구조

```text
ICRA_pipeline_v2/
├── configs/
│   ├── default.yaml
│   ├── fcm.yaml
│   ├── reward.yaml
│   └── evaluation.yaml
│
├── docs/
│   ├── README.md
│   ├── REVIEW_CHECKLIST.md
│   ├── stage_01_legacy_baseline.md
│   ├── ...
│   └── stage_16_end_to_end_evaluation.md
│
├── legacy/
│   ├── collect_demo_old.py
│   ├── feature_select_old.py
│   ├── phase_segment_old.py
│   └── fcm_old.py
│
├── reference_data/
│   └── demo_reference/
│       └── demo.hdf5
│
├── src/
│   └── bfd_pipeline/
│       ├── core/
│       │   ├── types.py
│       │   └── config.py
│       ├── data/
│       │   ├── demo_io.py
│       │   └── artifact_io.py
│       ├── envs/
│       │   ├── base_adapter.py
│       │   └── robosuite_adapter.py
│       ├── features/
│       │   ├── feature_functions.py
│       │   ├── feature_registry.py
│       │   ├── feature_engine.py
│       │   └── feature_profiler.py
│       ├── segmentation/
│       │   ├── boundary_evidence.py
│       │   ├── phase_segmenter.py
│       │   └── subgoal_inference.py
│       ├── consequence/
│       │   ├── perturbation_rollout.py
│       │   ├── fcm_dataset.py
│       │   ├── fcm_model.py
│       │   └── fcm_trainer.py
│       ├── degradation/
│       │   ├── hypothesis_generator.py
│       │   ├── candidate_generator.py
│       │   ├── degradation_search.py
│       │   └── simulator_validator.py
│       ├── preference/
│       │   ├── trajectory_dataset.py
│       │   ├── pair_generator.py
│       │   ├── reward_model.py
│       │   └── reward_trainer.py
│       ├── policy/
│       │   ├── bc.py
│       │   ├── reward_wrapper.py
│       │   └── improvement.py
│       └── evaluation/
│           ├── metrics.py
│           ├── evaluator.py
│           └── ablations.py
│
├── scripts/
├── tests/
├── artifacts/
└── pyproject.toml
```

---

# 5. 공통 데이터 구조

## 5.1 EpisodeData

```python
@dataclass
class EpisodeData:
    states: np.ndarray       # (T, state_dim)
    actions: np.ndarray      # (T, action_dim)
    model_xml: str | None
    env_info: dict
    success: bool | None
    source_path: str
    episode_key: str
```

Reference 환경에서는 일반적으로:

```text
states.shape  = (T, 77)
actions.shape = (T, 7)
```

단, 이 차원은 환경과 XML에 따라 달라질 수 있으므로 downstream에서 하드코딩하지 않는다.

## 5.2 RawTrajectory

```python
@dataclass
class RawTrajectory:
    time: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    actions: np.ndarray

    eef_pos: np.ndarray
    eef_quat: np.ndarray

    object_pos: np.ndarray
    object_quat: np.ndarray

    goal_pos: np.ndarray
    gripper_aperture: np.ndarray
    contact: np.ndarray

    actuator_force_or_torque_proxy: np.ndarray
    metadata: dict
```

## 5.3 FeatureTrajectory

```python
@dataclass
class FeatureTrajectory:
    values: np.ndarray   # (T, F)
    names: list[str]     # length F
    dt: float
    metadata: dict
```

## 5.4 FeatureProfile

```python
@dataclass
class FeatureProfile:
    name: str
    event_score: float
    trend_score: float
    plateau_score: float
    change_point_score: float
    endpoint_consistency: float
    cross_demo_consistency: float
    structural_score: float
    boundary_score: float
    confidence: float
```

Hard label은 가능하면 피한다.

```text
kind = progress
kind = event
```

대신:

```text
trend_score = 0.91
event_score = 0.08
boundary_score = 0.85
```

형태로 저장한다.

## 5.5 PhaseSegmentation

```python
@dataclass
class PhaseSegmentation:
    phase_ids: np.ndarray       # (T,)
    boundaries: list[int]       # [0, ..., T]
    boundary_scores: np.ndarray # (T,)
    labels: list[str]           # z0, z1, ...
    metadata: dict
```

## 5.6 PhaseSubgoal

```python
@dataclass
class PhaseSubgoal:
    phase_id: int
    change_features: list[dict]
    hold_features: list[dict]
    passive_features: list[str]
    endpoint_distribution: dict
    confidence: float
```

---

# 6. Stage별 코드 파이프라인

## Stage 1. Legacy Baseline

### 목적

기존 코드와 현재 결과를 보존한다.

```text
legacy/
reference demo
baseline manifest
legacy outputs
```

### 주요 검증

```text
legacy 코드 checksum
reference demo checksum
state/action shape
현재 feature name order
현재 phase boundary
현재 FCM input/output
```

## Stage 2. Project Scaffold와 Demo I/O

### 목적

HDF5를 안전하게 읽고 공통 dataclass로 변환한다.

### 주요 파일

```text
core/types.py
data/demo_io.py
data/artifact_io.py
```

### 핵심 검증

```python
assert len(states) == len(actions)
assert np.isfinite(states).all()
assert np.isfinite(actions).all()
```

## Stage 3. Robosuite Adapter와 Replay

### 목적

저장된 simulator state를 정확히 복원한다.

### 주요 기능

```text
environment build
state restore
raw signal extraction
actual goal extraction
success check
action layout
```

### 핵심 검증

```text
saved state
→ simulator restore
→ flattened state
→ original과 비교
```

Stage 3가 실패하면 이후 모든 결과를 신뢰할 수 없다.

## Stage 4. Feature Registry와 Engine

### 목적

사람이 feature 함수만 등록하도록 만든다.

```python
FEATURE_FUNCTIONS = {
    "eef_object_dist": eef_object_dist,
    "object_goal_dist": object_goal_dist,
    "contact": contact,
    "joint_energy": joint_energy,
}
```

### 금지

```text
progress=True
event=True
boundary=True
higher_is_worse=True
```

### Feature 예시

Structural 후보가 될 가능성이 있는 feature:

```text
eef_object_dist
object_goal_dist
object_height
contact
gripper_aperture
grasp_alignment
```

Passive consequence feature가 될 가능성이 있는 feature:

```text
eef_speed
eef_accel
eef_jerk
object_speed
action_magnitude
joint_energy
path_increment
object_slip
```

그러나 최종 역할은 사람이 지정하지 않는다.

## Stage 5. Automatic Feature Profiler

### 목적

Feature signal만 보고 다음 특성을 분석한다.

```text
event-like
trend-like
plateau-like
change-point
cross-demo consistency
structural confidence
boundary confidence
```

### 테스트용 synthetic signal

```text
monotonic decrease + plateau
monotonic increase + plateau
binary 0→1
binary 1→0
pure noise
sinusoidal signal
early outlier
aligned multi-demo event
random multi-demo event
```

Feature 이름을 보고 역할을 결정하지 않는다.

## Stage 6. Multivariate Phase Segmentation

### 목적

모든 structural feature의 boundary evidence를 결합해 phase를 자동 생성한다.

### 기본 알고리즘

```text
feature별 boundary evidence
        ↓
structural confidence 가중합
        ↓
temporal smoothing
        ↓
peak detection
        ↓
nearby boundary merge
        ↓
minimum phase length
        ↓
z0, z1, z2, ...
```

`Approach`, `Grasp`, `Transport`, `Place`를 알고리즘에 미리 넣지 않는다.

## Stage 7. Phase Subgoal Inference

### 목적

각 phase에서 feature 역할을 자동 추론한다.

### Change

```text
phase 안에서 일관된 방향으로 변화
start-end 차이가 큼
여러 demo에서 방향이 일치
```

### Hold

```text
phase 안에서 일정하게 유지
variance가 낮음
여러 demo에서 reference distribution이 안정적
```

### Passive

```text
phase 구조와 일관된 관계가 없음
confidence가 낮음
```

Milestone A는 여기까지다.

## Stage 8. Perturbation Rollout Dataset

### 목적

Demo action과 perturbed action의 consequence 차이를 simulator에서 측정한다.

One-step residual:

```math
y_{t+1}=\phi^{pert}_{t+1}-\phi^{demo}_{t+1}
```

H-step residual:

```math
y_{t+h}=\phi^{pert}_{t+h}-\phi^{demo}_{t+h}
```

### 핵심 검증

```text
delta = 0
→ residual ≈ 0
```

## Stage 9. Residual FCM

### 목적

현재 context와 action perturbation으로 future feature residual을 예측한다.

### 입력

```text
feature_t
action_t
delta_t
phase_id
phase_progress
goal_context
horizon
```

### 평가

```text
feature별 MAE
feature별 R²
phase별 R²
Spearman ranking
Top-K recall
held-out demo 성능
```

FCM의 핵심 지표는 Top-K recall이다.

Milestone B는 여기까지다.

## Stage 10. Degradation Hypothesis

### 목적

Subgoal로부터 열화 가설을 자동 생성한다.

Change feature:

```text
demo에서 관측된 변화 방향의 반대
```

Hold feature:

```text
reference distribution에서 이탈
```

Passive feature에서는 hypothesis를 생성하지 않는다.

## Stage 11. Candidate Search

### 목적

각 hypothesis에 대해 action perturbation 후보를 생성하고 FCM으로 Top-K를 고른다.

OSC_POSE 7-D 기준 action subspace:

```text
position : [0:3]
rotation : [3:6]
gripper  : [6:7]
```

단 adapter metadata에서 읽는다.

초기 구현은 phase 전체에 고정된 direction vector를 사용한다.

## Stage 12. Exact Simulator Validation

### 목적

FCM이 선별한 후보를 실제 simulator에서 검증한다.

각 candidate마다 독립적으로 lambda를 calibrate한다.

```text
candidate 1 → lambda_max_1
candidate 2 → lambda_max_2
candidate 3 → lambda_max_3
```

Nested hierarchy:

```text
lambda = 0
lambda = 0.25 * lambda_max
lambda = 0.50 * lambda_max
lambda = 0.75 * lambda_max
lambda = 1.00 * lambda_max
```

검증 항목:

```text
success
target feature degradation
monotonicity
gradedness
clipping
feasibility
```

실제 simulator score에서 `Original > Mild > Medium > Severe`가 확인된 family만 채택한다.

Milestone C는 여기까지다.

## Stage 13. Preference Dataset

### 목적

Validated family에서 trajectory와 pair를 생성한다.

같은 family 안에서만 생성한다.

```text
Original > Mild
Mild > Medium
Medium > Severe
Original > Severe
```

서로 다른 family 사이의 pair는 만들지 않는다.

## Stage 14. Preference Reward Learning

### 입력

```text
all consequence features
phase id
optional goal context
```

Preference probability:

```math
P(\tau_i \succ \tau_j)=\sigma\left(J_\theta(\tau_i)-J_\theta(\tau_j)\right)
```

이 단계에서 reward model이 다음을 preference로 학습한다.

```text
energy 방향
jerk 방향
slip 방향
feature 중요도
feature 상호작용
```

Milestone D는 여기까지다.

## Stage 15. Conservative Policy Improvement

순서:

```text
BC policy
        ↓
learned reward evaluation
        ↓
conservative fine-tuning
```

Retrospective phase segmentation은 online reward 계산에 바로 사용할 수 없으므로 다음 중 하나가 필요하다.

```text
online phase tracker
episode 후 replay-buffer relabeling
```

안전장치:

```text
behavior constraint
KL regularization
action regularization
task success guard
reward exploitation detection
```

## Stage 16. End-to-End Evaluation

비교 대상:

```text
suboptimal demonstrator
BC policy
improved policy
random/task baseline
```

평가 지표:

```text
success rate
final goal error
path length
energy proxy
jerk
action magnitude
contact stability
object slip
learned reward
```

Better-than-Demonstrator 기준:

```text
task success 유지 또는 개선
+
하나 이상의 efficiency metric 개선
```

필수 Ablation:

```text
random degradation vs structured degradation
FCM screening 없음 vs 있음
phase conditioning 없음 vs 있음
residual FCM vs non-residual FCM
automatic segmentation vs legacy/manual
quality feature 포함 vs 제거
```

---

# 7. 전체 Artifact 흐름

| Stage | 주요 Artifact |
|---|---|
| 1 | `baseline_manifest.json` |
| 2 | `demo_inspection.json` |
| 3 | `raw_trajectory.npz` |
| 4 | `features.npz` |
| 5 | `feature_profiles.json` |
| 6 | `phase_seg.npz` |
| 7 | `subgoals.json` |
| 8 | `fcm_dataset.hdf5` |
| 9 | `fcm_model.pt` |
| 10 | `degradation_hypotheses.json` |
| 11 | `screened_candidates.json` |
| 12 | `validated_families.json` |
| 13 | `preference_trajectories.hdf5` |
| 14 | `reward_model.pt` |
| 15 | `improved_policy.pt` |
| 16 | `summary_metrics.csv` |

모든 artifact에는 다음 metadata를 포함한다.

```text
schema version
feature name order
environment
robot
controller
action dimension
control frequency
random seed
source demo checksum
git commit
```

---

# 8. Milestone 구성

## Milestone A — Demo Understanding

```text
Stage 1 ~ Stage 7
```

결과:

```text
feature 함수만 정의
→ 자동 feature profiling
→ 자동 phase segmentation
→ 자동 subgoal inference
```

## Milestone B — Consequence Prediction

```text
Stage 8 ~ Stage 9
```

결과:

```text
simulator perturbation dataset
→ residual FCM
```

## Milestone C — Structured Degradation

```text
Stage 10 ~ Stage 12
```

결과:

```text
subgoal-based hypothesis
→ FCM screening
→ simulator-validated hierarchy
```

## Milestone D — Preference Reward

```text
Stage 13 ~ Stage 14
```

결과:

```text
preference dataset
→ phase-conditioned reward
```

## Milestone E — Better-than-Demonstrator Policy

```text
Stage 15 ~ Stage 16
```

결과:

```text
conservative policy improvement
→ end-to-end evaluation
```

---

# 9. 핵심 검증 Gate

## Gate A — State Restore

```text
saved state
→ restore
→ flattened state
→ original과 오차가 tolerance 이하
```

실패하면 중단한다.

## Gate B — Feature Function Only

```text
feature registry에 이름과 함수만 존재
```

추가 metadata가 있으면 설계를 다시 검토한다.

## Gate C — Automatic Segmentation

```text
synthetic signal test 통과
random noise에 안정적
phase 수가 폭발하지 않음
```

## Gate D — Zero Perturbation

```text
delta = 0
→ feature residual ≈ 0
```

실패하면 FCM 학습 금지.

## Gate E — FCM Screening

```text
actual good candidate가 FCM Top-K에 포함
```

Random baseline보다 나아야 한다.

## Gate F — Simulator Validation

```text
실제 degradation score가 nested lambda에 따라 단조
```

통과한 family만 preference dataset으로 보낸다.

## Gate G — Reward Generalization

```text
held-out demo
held-out family
```

에서 chance보다 높은 preference accuracy를 보여야 한다.

## Gate H — Policy Safety

```text
learned reward 증가
but
task success 붕괴
```

가 발생하면 reward exploitation으로 간주한다.

---

# 10. 개발 운영 규칙

각 Stage는 다음 순서로 진행한다.

```text
Stage 문서 전달
        ↓
Coding Agent 계획 확인
        ↓
구현
        ↓
pytest
        ↓
artifact 생성
        ↓
그래프 직접 확인
        ↓
git diff 확인
        ↓
commit/tag
        ↓
다음 Stage
```

권장 Git 이름:

```text
stage-01-legacy-baseline
stage-02-demo-io
stage-03-robosuite-adapter
stage-04-feature-engine
...
```

---

# 11. 최종 파이프라인 한 줄 요약

```text
사람이 feature 계산 함수만 정의
→ 알고리즘이 feature 구조와 phase/subgoal을 자동 추론
→ FCM이 subgoal degradation action 후보를 선별
→ simulator가 실제 degradation hierarchy를 검증
→ preference reward가 전체 feature의 방향과 중요도를 학습
→ 보수적인 policy improvement로 demonstrator를 능가
```

---

# 12. 현재 구현 우선순위

현재 가장 먼저 완성해야 하는 범위는 전체 pipeline이 아니다.

```text
Stage 1
→ Stage 2
→ Stage 3
→ Stage 4
```

먼저 다음 데이터 흐름을 확실하게 만든다.

```text
demo.hdf5
→ EpisodeData
→ RawTrajectory
→ FeatureTrajectory
```

그다음:

```text
Stage 5
→ Stage 6
→ Stage 7
```

을 통해 Milestone A를 완성한다.

FCM은 Milestone A가 안정적으로 검증된 이후에 구현한다.
