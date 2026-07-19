# BfD 코드 파이프라인 구현 설계도 v3
## Feature-Function-Only Automatic Phase Segmentation + Residual FCM + Simulator-Validated Structured Degradation

---

# 0. 문서의 상태와 범위

이 문서는 단순한 아이디어 정리가 아니라, Claude Code 또는 다른 coding agent가 각 Stage를 순서대로 구현할 수 있도록 작성한 **구현 설계도**다.

본 문서에서 파이프라인은 다음 두 구역으로 나눈다.

## 0.1 현재 고정된 핵심 범위

아래 범위는 현재 프로젝트의 주된 구현 목표이며, 본 문서에서 입출력, 알고리즘, 검증 조건까지 고정한다.

```text
Stage 1  ~ Stage 7  : Demonstration Understanding
Stage 8  ~ Stage 9  : Forward Consequence Model
Stage 10 ~ Stage 12 : Structured Degradation
```

즉 다음 흐름은 구현 대상으로 확정한다.

```text
suboptimal demonstrations
        ↓
feature trajectory
        ↓
automatic phase segmentation
        ↓
phase subgoal inference
        ↓
perturbation consequence dataset
        ↓
residual FCM
        ↓
subgoal degradation hypothesis
        ↓
FCM candidate screening
        ↓
exact simulator validation
        ↓
validated degradation families
```

## 0.2 아직 연구적으로 열려 있는 범위

아래 범위는 인터페이스만 정의하고, 최종 알고리즘은 아직 고정하지 않는다.

```text
Stage 13 : Preference Dataset
Stage 14 : Reward Learning
Stage 15 : Better-than-Demonstrator Policy Improvement
Stage 16 : End-to-End Evaluation
```

현재 확정된 결과물은 `ValidatedDegradationFamily`까지다.

따라서 Stage 12가 끝난 시점에 다음 주장은 가능하다.

> 사람의 suboptimal demonstration을 자동으로 phase와 subgoal로 분해하고, 각 phase의 subgoal을 의도적으로 열화시키는 구조화된 action perturbation을 생성한 뒤, FCM과 simulator를 이용하여 성공을 유지하면서 점진적으로 나빠지는 trajectory family를 생성하였다.

반면 다음 주장은 Stage 13 이후가 검증되기 전에는 하지 않는다.

> 학습된 policy가 demonstrator를 능가한다.

---

# 1. 프로젝트 목표

## 1.1 핵심 목표

성공했지만 비효율적인 demonstration으로부터 단순 모방이 아니라, demonstration의 내부 구조를 다음과 같이 추출한다.

```text
어떤 phase들이 존재하는가?
각 phase에서 무엇이 변화해야 하는가?
각 phase가 끝날 때 무엇이 유지되어야 하는가?
어떤 action perturbation이 그 subgoal을 열화시키는가?
그 열화가 simulator에서 실제로 점진적이고 유효한가?
```

## 1.2 현재 1차 대상 환경

초기 구현과 검증 대상은 다음으로 제한한다.

```text
Simulator       : robosuite + MuJoCo
Task            : single-object Pick-and-Place
Robot           : single-arm robot
Controller      : OSC_POSE
Action layout   : [dx, dy, dz, droll, dpitch, dyaw, gripper]
Action dimension: 7
Demo type       : successful but intentionally suboptimal teleoperation
```

다른 task 또는 robot으로 일반화하기 전에 위 설정에서 모든 Gate를 통과해야 한다.

## 1.3 요구되는 demonstration 수

자동 segmentation과 cross-demo consistency를 사용하므로 단일 demonstration만으로 최종 모델을 확정하지 않는다.

권장 기본값:

```text
minimum usable demos   : 10
recommended demos      : 20 ~ 50
held-out validation    : 전체 demo의 20%
all demos              : task success = True
quality variation      : detour, pause, over-correction, inefficient path 포함 가능
```

단일 demonstration 모드는 디버깅용으로만 지원하며, 이 경우 cross-demo 관련 confidence를 0으로 두고 결과를 `debug_only=True`로 저장한다.

---

# 2. 고정 설계 원칙

## 2.1 사람은 feature 계산 함수만 정의한다

사람이 등록하는 것은 feature 이름과 계산 함수다.

```python
FEATURE_FUNCTIONS = {
    "eef_object_dist": compute_eef_object_dist,
    "object_goal_dist": compute_object_goal_dist,
    "object_height": compute_object_height,
    "gripper_aperture": compute_gripper_aperture,
    "contact": compute_contact,
    "eef_speed": compute_eef_speed,
    "eef_jerk": compute_eef_jerk,
    "joint_energy": compute_joint_energy,
    "path_increment": compute_path_increment,
    "object_slip": compute_object_slip,
}
```

사람이 직접 지정하지 않는 정보:

```text
progress feature 여부
event feature 여부
quality feature 여부
boundary feature 여부
higher-is-better / higher-is-worse
phase 이름
phase 개수
phase 경계
subgoal
열화 action 방향
```

## 2.2 이름 기반 추론을 금지한다

다음은 금지한다.

```python
if "contact" in feature_name:
    feature_role = "event"

if "energy" in feature_name:
    higher_is_worse = True
```

feature의 역할은 이름이 아니라 signal의 통계적 형태에서 추론한다.

## 2.3 Structural feature와 Passive consequence feature를 분리한다

모든 feature는 계산하지만, phase segmentation과 degradation hypothesis에는 신뢰도 높은 structural feature만 사용한다.

```text
Structural feature
- phase 경계 또는 subgoal을 설명할 가능성이 높은 feature
- 반복 demo에서 변화 시점과 방향이 일관됨
- event, trend, plateau, endpoint consistency가 높음

Passive consequence feature
- energy, jerk, path length처럼 결과의 질을 나타낼 수 있으나
  phase 구조를 직접 결정하기에는 불안정한 feature
- degradation rollout에서는 모두 저장
- 최종 reward 단계에서 활용 가능
```

Structural 여부는 hard label 하나로 결정하지 않고 `structural_score`와 uncertainty로 저장한다.

## 2.4 FCM은 최종 판정기가 아니다

FCM의 역할은 수많은 perturbation 후보 중 simulator에서 실행할 후보를 줄이는 것이다.

```text
Candidate 1, 2, 3, ..., N
        ↓
FCM consequence prediction
        ↓
subgoal degradation score 계산
        ↓
uncertainty 및 diversity 반영
        ↓
Top-K candidates
        ↓
exact simulator rollout
        ↓
최종 채택 또는 폐기
```

FCM 예측만으로 preference label이나 degradation family를 확정하지 않는다.

## 2.5 모든 degradation level은 가능한 한 task success를 유지한다

본 프로젝트의 목표는 단순한 실패 trajectory 생성이 아니다.

원칙:

```text
Original : 성공
Mild     : 성공 유지 + subgoal 약화
Medium   : 성공 유지 + subgoal 더 약화
Severe   : 성공 유지 가능한 최대 열화
```

성공을 즉시 깨뜨리는 perturbation은 일반적인 degradation family로 채택하지 않는다.

별도의 failure-negative dataset이 필요할 경우 Stage 13 이후의 선택 사항으로 분리한다.

## 2.6 모든 Stage는 독립적으로 중단 가능해야 한다

각 Stage는 다음을 남겨야 한다.

```text
1. typed artifact
2. configuration snapshot
3. deterministic test
4. quantitative metrics
5. visualization or inspection report
6. PASS / FAIL gate result
```

이전 Stage가 FAIL이면 다음 Stage를 실행하지 않는다.

---

# 3. 전체 파이프라인

```text
[Stage 1] Demo Collection / Legacy Preservation
        ↓ demo.hdf5

[Stage 2] Demo Loader
        ↓ EpisodeData

[Stage 3] Robosuite State Replay Adapter
        ↓ RawTrajectory

[Stage 4] Feature Engine
        ↓ FeatureTrajectory

[Stage 5] Automatic Feature Profiling
        ↓ FeatureProfileSet

[Stage 6] Robust Multi-Demo Phase Segmentation
        ↓ PhaseSegmentationSet

[Stage 7] Phase Subgoal Inference
        ↓ CanonicalPhaseModel + PhaseSubgoalSet

[Stage 8] Perturbation Rollout Collection
        ↓ FCMDataset

[Stage 9] Residual FCM Training
        ↓ FCMEnsemble

[Stage 10] Degradation Hypothesis Generation
        ↓ DegradationHypothesisSet

[Stage 11] Candidate Generation and FCM Screening
        ↓ ScreenedCandidateSet

[Stage 12] Exact Simulator Validation
        ↓ ValidatedDegradationFamilySet

--------------------------------------------------
현재 확정 범위 종료
--------------------------------------------------

[Stage 13] Preference Construction       [Provisional]
[Stage 14] Reward Learning               [Provisional]
[Stage 15] Policy Improvement            [Provisional]
[Stage 16] Better-than-Demo Evaluation   [Provisional]
```

---

# 4. 권장 프로젝트 구조

```text
ROBOT_BFD/
├── configs/
│   ├── base.yaml
│   ├── data.yaml
│   ├── features.yaml
│   ├── segmentation.yaml
│   ├── fcm.yaml
│   ├── degradation.yaml
│   └── provisional_bfd.yaml
│
├── docs/
│   ├── PIPELINE_BLUEPRINT.md
│   ├── IMPLEMENTATION_STATUS.md
│   ├── REVIEW_CHECKLIST.md
│   ├── stage_01_demo.md
│   ├── stage_02_demo_io.md
│   ├── stage_03_replay.md
│   ├── stage_04_features.md
│   ├── stage_05_profiling.md
│   ├── stage_06_segmentation.md
│   ├── stage_07_subgoals.md
│   ├── stage_08_fcm_dataset.md
│   ├── stage_09_fcm.md
│   ├── stage_10_hypothesis.md
│   ├── stage_11_candidate_search.md
│   └── stage_12_sim_validation.md
│
├── reference_data/
│   ├── demos/
│   │   ├── demo_000.hdf5
│   │   ├── demo_001.hdf5
│   │   └── ...
│   └── manifests/
│
├── src/
│   └── bfd_pipeline/
│       ├── core/
│       │   ├── types.py
│       │   ├── schemas.py
│       │   ├── config.py
│       │   ├── seed.py
│       │   └── logging.py
│       │
│       ├── data/
│       │   ├── demo_io.py
│       │   ├── artifact_io.py
│       │   └── split.py
│       │
│       ├── envs/
│       │   ├── base_adapter.py
│       │   ├── robosuite_adapter.py
│       │   ├── state_restore.py
│       │   └── action_layout.py
│       │
│       ├── features/
│       │   ├── feature_functions.py
│       │   ├── registry.py
│       │   ├── engine.py
│       │   ├── normalization.py
│       │   └── profiler.py
│       │
│       ├── segmentation/
│       │   ├── evidence.py
│       │   ├── changepoint.py
│       │   ├── objective.py
│       │   ├── segmenter.py
│       │   ├── sequence_alignment.py
│       │   ├── stability.py
│       │   └── subgoal_inference.py
│       │
│       ├── consequence/
│       │   ├── rollout_sampler.py
│       │   ├── dataset.py
│       │   ├── model.py
│       │   ├── ensemble.py
│       │   ├── trainer.py
│       │   └── evaluation.py
│       │
│       ├── degradation/
│       │   ├── hypothesis.py
│       │   ├── candidate.py
│       │   ├── parameterization.py
│       │   ├── scoring.py
│       │   ├── screening.py
│       │   ├── lambda_search.py
│       │   ├── simulator_validator.py
│       │   └── family_builder.py
│       │
│       ├── provisional_bfd/
│       │   ├── preference.py
│       │   ├── reward.py
│       │   └── improvement.py
│       │
│       └── visualization/
│           ├── feature_plots.py
│           ├── segmentation_plots.py
│           ├── fcm_plots.py
│           └── degradation_plots.py
│
├── scripts/
│   ├── run_stage_01.py
│   ├── run_stage_02.py
│   ├── ...
│   ├── run_stage_12.py
│   └── run_pipeline_to_degradation.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── synthetic/
│   └── regression/
│
├── artifacts/
│   ├── stage_01/
│   ├── ...
│   └── stage_12/
│
├── pyproject.toml
└── README.md
```

---

# 5. 공통 데이터 계약

모든 dataclass는 schema version과 provenance를 포함한다.

## 5.1 ArtifactMetadata

```python
@dataclass
class ArtifactMetadata:
    schema_version: str
    created_at: str
    git_commit: str | None
    random_seed: int
    config_hash: str
    source_artifact_ids: list[str]
    environment_name: str
    robot_name: str
    controller_name: str
    control_frequency: float
    feature_name_order: list[str]
```

## 5.2 EpisodeData

```python
@dataclass
class EpisodeData:
    episode_id: str
    states: np.ndarray          # (T, state_dim)
    actions: np.ndarray         # (T, action_dim)
    model_xml: str | None
    env_info: dict
    success: bool
    source_path: str
    metadata: ArtifactMetadata
```

검증:

```python
assert states.ndim == 2
assert actions.ndim == 2
assert len(states) == len(actions)
assert np.isfinite(states).all()
assert np.isfinite(actions).all()
```

## 5.3 RawTrajectory

```python
@dataclass
class RawTrajectory:
    episode_id: str
    time: np.ndarray
    simulator_states: np.ndarray
    actions: np.ndarray

    qpos: np.ndarray
    qvel: np.ndarray
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    object_pos: np.ndarray
    object_quat: np.ndarray
    goal_context: dict
    gripper_aperture: np.ndarray
    contact_signals: dict[str, np.ndarray]
    actuator_proxy: np.ndarray | None

    success: bool
    metadata: ArtifactMetadata
```

`goal_context`는 단일 `goal_pos`로 하드코딩하지 않는다.

초기 Pick-and-Place에서는 다음을 포함할 수 있다.

```python
goal_context = {
    "type": "target_region",
    "target_pos": np.ndarray(shape=(3,)),
    "target_quat": np.ndarray(shape=(4,)) | None,
    "position_tolerance": float,
    "orientation_tolerance": float | None,
}
```

## 5.4 FeatureTrajectory

```python
@dataclass
class FeatureTrajectory:
    episode_id: str
    raw_values: np.ndarray          # (T, F)
    normalized_values: np.ndarray   # (T, F)
    names: list[str]
    dt: float
    valid_mask: np.ndarray          # (T, F)
    metadata: ArtifactMetadata
```

정규화는 train demo에서 계산한 robust statistics를 사용한다.

```text
center = median
scale  = max(IQR / 1.349, epsilon)
normalized = clip((x - center) / scale, -clip_value, clip_value)
```

기본값:

```text
epsilon    = 1e-6
clip_value = 10.0
```

## 5.5 FeatureProfile

```python
@dataclass
class FeatureProfile:
    name: str

    binary_score: float
    event_score: float
    trend_score: float
    plateau_score: float
    changepoint_score: float
    endpoint_consistency: float
    cross_demo_timing_consistency: float
    cross_demo_direction_consistency: float
    noise_robustness: float

    structural_score: float
    structural_uncertainty: float
    confidence: float
```

## 5.6 BoundaryEvidence

```python
@dataclass
class BoundaryEvidence:
    episode_id: str
    feature_names: list[str]
    per_feature_evidence: np.ndarray   # (T, F)
    combined_evidence: np.ndarray      # (T,)
    reliability_weights: np.ndarray    # (F,)
    metadata: dict
```

## 5.7 PhaseSegmentation

```python
@dataclass
class PhaseSegmentation:
    episode_id: str
    boundaries: list[int]           # [0, b1, ..., T]
    phase_ids: np.ndarray           # (T,)
    boundary_scores: np.ndarray     # (T,)
    objective_value: float
    bootstrap_support: list[float]
    leave_one_feature_out_shift: list[float]
    confidence: float
    debug_only: bool
    metadata: ArtifactMetadata
```

## 5.8 SegmentDescriptor

```python
@dataclass
class SegmentDescriptor:
    episode_id: str
    local_phase_id: int
    start: int
    end: int

    mean: np.ndarray
    std: np.ndarray
    delta: np.ndarray
    slope: np.ndarray
    endpoint: np.ndarray
    event_occupancy: np.ndarray
    duration_normalized: float
```

## 5.9 CanonicalPhaseModel

```python
@dataclass
class CanonicalPhaseModel:
    canonical_phase_count: int
    canonical_labels: list[str]             # z0, z1, ...
    phase_descriptor_centers: np.ndarray
    phase_descriptor_scales: np.ndarray
    sequence_transition_matrix: np.ndarray
    demo_alignment_paths: dict[str, list[int]]
    confidence: float
```

## 5.10 PhaseSubgoal

```python
@dataclass
class PhaseSubgoal:
    canonical_phase_id: int

    change_features: list[dict]
    hold_features: list[dict]
    passive_features: list[str]

    endpoint_center: np.ndarray
    endpoint_scale: np.ndarray
    completion_score_definition: dict
    confidence: float
```

`change_features` 항목 예시:

```python
{
    "name": "eef_object_dist",
    "demo_direction": -1,              # demo에서 감소
    "median_delta": -2.31,             # normalized scale
    "direction_consistency": 0.92,
    "importance": 0.78,
}
```

`hold_features` 항목 예시:

```python
{
    "name": "contact",
    "reference_center": 0.98,
    "reference_scale": 0.05,
    "hold_consistency": 0.94,
    "importance": 0.85,
}
```

## 5.11 FCMSample

```python
@dataclass
class FCMSample:
    episode_id: str
    timestep: int
    canonical_phase_id: int
    phase_progress: float

    feature_t: np.ndarray
    action_t: np.ndarray
    perturbation: np.ndarray
    perturbation_window: int
    goal_embedding: np.ndarray

    horizons: np.ndarray
    residual_targets: np.ndarray       # (H, F)
    feasibility_flags: dict
```

## 5.12 FCMEnsemblePrediction

```python
@dataclass
class FCMEnsemblePrediction:
    residual_mean: np.ndarray
    residual_std: np.ndarray
    member_predictions: np.ndarray
```

## 5.13 DegradationHypothesis

```python
@dataclass
class DegradationHypothesis:
    hypothesis_id: str
    canonical_phase_id: int
    target_type: str                   # change_opposition | hold_violation
    target_feature_names: list[str]
    target_definition: dict
    confidence: float
```

## 5.14 PerturbationCandidate

```python
@dataclass
class PerturbationCandidate:
    candidate_id: str
    hypothesis_id: str
    canonical_phase_id: int

    start_fraction: float
    duration_fraction: float
    action_subspace: str
    direction: np.ndarray              # unit vector, action_dim

    predicted_degradation: float | None
    predicted_uncertainty: float | None
    screening_score: float | None
```

## 5.15 ValidatedDegradationFamily

```python
@dataclass
class ValidatedDegradationFamily:
    family_id: str
    candidate: PerturbationCandidate
    lambda_max: float
    lambda_levels: list[float]

    trajectories: list[dict]
    degradation_scores: list[float]
    task_success: list[bool]
    feasibility: list[bool]

    monotonicity_score: float
    gradedness_score: float
    recovery_score: float
    confidence: float
```

---

# 6. Stage 1 — Demonstration Collection and Legacy Preservation

## 6.1 목적

기존 teleoperation 코드와 reference demonstration을 보존하고, 새 파이프라인의 입력을 고정한다.

## 6.2 요구사항

각 demonstration은 다음을 만족해야 한다.

```text
1. task success = True
2. state/action length 일치
3. NaN/Inf 없음
4. model XML 또는 동일 환경 복원 정보 존재
5. action layout 기록
6. control frequency 기록
7. random seed 또는 collection session 기록
```

## 6.3 suboptimal demonstration의 허용 범위

허용:

```text
우회 경로
over-correction
짧은 pause
불필요한 회전
과도한 lifting
느린 transport
재정렬
```

초기 데이터셋에서 제외:

```text
최종 task 실패
object가 workspace 밖으로 이탈
simulator reset 또는 numerical instability
조작자의 action 기록 누락
비정상적인 episode truncation
```

## 6.4 산출물

```text
artifacts/stage_01/demo_manifest.json
reference_data/demos/demo_XXX.hdf5
```

## 6.5 Gate 1

```text
PASS 조건
- usable successful demos >= 10
- 모든 demo의 replay metadata 존재
- action dimension 및 controller가 dataset 내부에서 일관됨
```

---

# 7. Stage 2 — Demo Loader

## 7.1 목적

HDF5 내부 구조 차이를 downstream에서 제거하고, 모든 episode를 `EpisodeData`로 변환한다.

## 7.2 동작

```text
HDF5 탐색
→ episode group 검출
→ states/actions/model_xml/env_info 읽기
→ dtype 통일(float32 또는 float64 정책 고정)
→ 유효성 검사
→ EpisodeData 생성
```

## 7.3 오류 정책

```text
critical error
- states/actions 없음
- 길이 불일치
- NaN/Inf
- empty episode
→ 해당 episode reject

warning
- success flag 없음
- optional metadata 없음
→ adapter에서 복원 후 success 재계산
```

## 7.4 산출물

```text
artifacts/stage_02/episodes_index.json
artifacts/stage_02/demo_inspection.json
```

## 7.5 Gate 2

모든 accepted episode가 deterministic하게 동일 checksum과 shape를 반환해야 한다.

---

# 8. Stage 3 — Robosuite State Replay Adapter

## 8.1 목적

저장된 simulator state를 정확히 복원하고 feature 계산에 필요한 raw signal을 추출한다.

## 8.2 핵심 원칙

관측값만으로 simulator state를 추정하지 않는다.

가능하면 저장된 flattened simulator state를 직접 복원한다.

```text
saved simulator state
→ env.sim.set_state_from_flattened(...)
→ env.sim.forward()
→ raw signal extraction
```

## 8.3 동일 state 복원 검증

각 episode에서 무작위 timestep을 선택한다.

```text
original saved state s_t
→ restore
→ flatten restored state ŝ_t
→ error = max_abs(s_t - ŝ_t)
```

기본 tolerance:

```text
float64 state: 1e-8 ~ 1e-6
float32 state: 1e-5 ~ 1e-4
```

환경에 맞춰 config로 설정하되, tolerance를 초과하면 Stage 3 FAIL이다.

## 8.4 goal context 추출

초기 Pick-and-Place에서는 target region 또는 target object pose를 adapter가 제공한다.

다른 task 확장 시 `goal_context`를 task-specific adapter가 생성한다.

feature 함수는 robosuite 내부 private field를 직접 읽지 않고 adapter interface를 통해 접근한다.

## 8.5 산출물

```text
artifacts/stage_03/raw_trajectories/*.npz
artifacts/stage_03/replay_validation.json
```

## 8.6 Gate 3 — State Restore

```text
PASS 조건
- sampled restore error <= tolerance
- replay 후 object/eef pose가 저장 당시 값과 일치
- success 재계산 결과가 dataset label과 일치하거나 차이가 설명됨
```

Stage 3가 실패하면 이후 모든 Stage를 중단한다.

---

# 9. Stage 4 — Feature Engine

## 9.1 목적

등록된 feature 계산 함수를 모든 trajectory에 동일한 순서로 적용한다.

## 9.2 feature 함수 인터페이스

```python
FeatureFunction = Callable[[RawTrajectory, int, FeatureContext], float]
```

각 함수는 다음을 보장한다.

```text
- 같은 입력에 같은 출력
- side effect 없음
- feature name order 고정
- invalid value는 NaN을 조용히 반환하지 않고 valid_mask에 기록
```

## 9.3 파생량 계산

속도, 가속도, jerk는 control frequency를 반영한다.

```text
speed: finite difference + optional smoothing
accel: derivative of speed
jerk : derivative of accel
```

smoothing은 feature별 임의 설정이 아니라 공통 config를 사용한다.

권장 기본값:

```text
filter               : Savitzky-Golay or Gaussian
window duration      : 0.10 ~ 0.20 sec
polynomial order     : 2 or 3
edge handling        : reflection or nearest
```

원본 raw feature와 smoothed feature를 모두 저장할 수 있지만, segmentation 입력은 normalized smoothed signal을 사용한다.

## 9.4 정규화

train split 전체에서 robust scaler를 fit한다.

validation/test demo의 통계는 scaler 계산에 사용하지 않는다.

## 9.5 산출물

```text
artifacts/stage_04/features/*.npz
artifacts/stage_04/feature_scaler.json
artifacts/stage_04/feature_summary.csv
```

## 9.6 Gate 4

```text
PASS 조건
- feature order가 모든 demo에서 동일
- invalid 비율이 feature별 허용치를 넘지 않음
- constant feature와 near-constant feature가 보고됨
- synthetic geometry test 통과
```

---

# 10. Stage 5 — Automatic Feature Profiling

## 10.1 목적

feature 이름을 사용하지 않고 각 signal의 동적 특성과 phase 구조 기여도를 계산한다.

## 10.2 입력

```text
여러 demo의 normalized FeatureTrajectory
```

## 10.3 feature별 분석 항목

### A. Binary-like score

feature가 두 개의 안정된 상태를 갖는지 평가한다.

예:

```text
0 근처와 1 근처에 밀집
중간값 체류 시간이 짧음
transition 횟수가 적음
```

### B. Event score

짧은 시간 안에 상태가 급변하고 그 이후 변화가 유지되는지 평가한다.

```text
large local derivative
+
pre/post window mean difference
+
post-event persistence
```

### C. Trend score

구간 내에서 일관된 방향으로 변화하는지 평가한다.

```text
Spearman(|rho|)
+
robust slope magnitude
+
sign consistency across windows
```

### D. Plateau score

변화 후 낮은 derivative 상태가 유지되는지 평가한다.

### E. Change-point score

해당 feature를 두 구간으로 나눌 때 within-segment fit이 얼마나 개선되는지 평가한다.

### F. Endpoint consistency

성공 demo들의 마지막 구간에서 feature가 비슷한 분포로 수렴하는지 평가한다.

### G. Cross-demo direction consistency

각 demo에서 주요 변화의 부호가 얼마나 일치하는지 평가한다.

### H. Cross-demo timing consistency

절대 timestep이 아니라 normalized progress 또는 segment alignment 후 변화 순서가 얼마나 일치하는지 평가한다.

### I. Noise robustness

다음 perturbation에 profiler score가 얼마나 안정적인지 측정한다.

```text
small Gaussian measurement noise
temporal jitter
random frame dropout interpolation
filter window variation
```

## 10.4 structural score

기본 구조:

```text
structural_score =
    w_event      * event_score
  + w_trend      * trend_score
  + w_plateau    * plateau_score
  + w_cp         * changepoint_score
  + w_endpoint   * endpoint_consistency
  + w_direction  * cross_demo_direction_consistency
  + w_timing     * cross_demo_timing_consistency
  + w_robust     * noise_robustness
```

초기 가중치는 config에 명시한다.

가중치 자체를 feature 이름에 따라 다르게 주지 않는다.

## 10.5 uncertainty

bootstrap resampling으로 score 분산을 측정한다.

```text
N bootstrap samples
→ demo subset 재추출
→ profiler 재계산
→ structural score distribution
```

기본값:

```text
bootstrap_samples = 100
```

## 10.6 Structural feature 사용 조건

다음 조건을 모두 만족해야 Stage 6의 주요 evidence에 포함한다.

```text
structural_score >= threshold
confidence >= threshold
noise_robustness >= threshold
```

threshold는 validation demo에서 고정하며 테스트 demo에 맞춰 조정하지 않는다.

## 10.7 synthetic tests

반드시 포함:

```text
monotonic decrease + plateau
monotonic increase + plateau
binary 0→1
binary 1→0
piecewise linear signal
multi-change signal
pure noise
sinusoidal signal
early constant signal
early outlier
late outlier
same event with temporal jitter across demos
random event timing across demos
```

## 10.8 산출물

```text
artifacts/stage_05/feature_profiles.json
artifacts/stage_05/profile_bootstrap.npz
artifacts/stage_05/profile_report.html or png
```

## 10.9 Gate 5

```text
PASS 조건
- synthetic event/trend signals가 noise보다 높은 structural score
- pure noise가 structural feature로 선택되지 않음
- bootstrap confidence가 설정 기준 이상
- feature 이름을 바꿔도 결과가 동일
```

---

# 11. Stage 6 — Robust Multi-Demo Phase Segmentation

이 Stage는 프로젝트의 핵심이다.

단순히 `np.argmax`, 단일 knee, 단일 peak를 사용하지 않는다.

## 11.1 목표

각 demonstration을 의미 있는 연속 구간으로 나누되 다음을 만족한다.

```text
1. 사람의 phase label이 필요 없음
2. phase 개수를 미리 지정하지 않음
3. 특정 feature 하나에 의존하지 않음
4. suboptimal detour와 pause에 지나치게 민감하지 않음
5. 여러 demo에서 유사한 phase sequence가 나타남
6. 작은 noise나 feature 제거로 경계가 크게 흔들리지 않음
```

## 11.2 전체 알고리즘

```text
Step 1. 각 feature의 local boundary evidence 계산
Step 2. feature reliability로 evidence 가중 결합
Step 3. 각 demo에서 candidate boundary pool 생성
Step 4. dynamic programming으로 최적 segmentation 계산
Step 5. 각 segment descriptor 계산
Step 6. demo 간 segment sequence alignment
Step 7. canonical phase count 및 canonical sequence 추론
Step 8. unstable boundary 제거 또는 merge
Step 9. bootstrap / feature ablation 안정성 검증
Step 10. 최종 PhaseSegmentationSet 저장
```

## 11.3 Step 1 — Local boundary evidence

각 feature마다 timestep `t`에서 다음 evidence를 계산한다.

### A. Mean-shift evidence

`t` 전후 window의 robust mean 차이.

### B. Slope-change evidence

`t` 전후 robust slope의 변화.

### C. Variance-change evidence

변화 구간에서 hold 구간으로 바뀌는 경우를 잡기 위한 분산 변화.

### D. Event-transition evidence

binary-like 또는 event-like feature의 상태 전환.

### E. Plateau-entry evidence

변화량이 크던 signal이 안정 상태로 진입하는 지점.

### F. Multiscale change-point evidence

여러 window size에서 change-point score를 계산하고 합친다.

기본 window 후보:

```text
0.10 sec
0.20 sec
0.40 sec
0.80 sec
```

환경 control frequency에 따라 frame 수로 변환한다.

## 11.4 Step 2 — Evidence 결합

각 feature의 reliability weight:

```text
reliability_f =
    structural_score_f
  * confidence_f
  * noise_robustness_f
```

combined evidence:

```text
E(t) = robust_weighted_mean_f(
         reliability_f * evidence_f(t)
       )
```

단순 합보다 outlier feature에 강한 trimmed mean 또는 Huber aggregation을 권장한다.

## 11.5 Step 3 — Candidate boundary pool

`E(t)`를 smoothing한 뒤 local maximum을 찾는다.

단, 이 단계의 peak는 최종 boundary가 아니라 후보일 뿐이다.

후보 생성 조건:

```text
prominence >= min_prominence
minimum distance >= min_boundary_gap
boundary cannot be within edge_margin from episode start/end
```

항상 `0`과 `T`를 포함한다.

## 11.6 Step 4 — Dynamic programming segmentation

최종 boundary 집합 `B`는 다음 목적함수를 최소화한다.

```text
J(B) =
    segment_fit_cost(B)
  - boundary_evidence_reward(B)
  + complexity_penalty(|B|)
  + short_segment_penalty(B)
```

### segment_fit_cost

각 segment 내부에서 feature trajectory가 다음 중 하나로 설명되는 정도를 계산한다.

```text
constant
linear trend
single transition + hold
```

세 모델 중 최소 robust regression cost를 사용한다.

### boundary_evidence_reward

높은 combined evidence 위치를 선택하면 cost가 감소한다.

### complexity_penalty

phase 수 폭발을 방지한다.

```text
penalty = beta * number_of_internal_boundaries
```

### short_segment_penalty

최소 phase duration보다 짧은 segment는 매우 큰 penalty를 부여한다.

기본값:

```text
min_phase_duration_sec = 0.25 ~ 0.50
max_phase_count        = 10
```

초기 Pick-and-Place에서 `max_phase_count`는 안전장치일 뿐, 목표 phase 수를 의미하지 않는다.

## 11.7 Step 5 — Segment descriptor

각 segment를 다음 통계로 표현한다.

```text
normalized duration
feature mean
feature std
start-to-end delta
robust slope
endpoint value
event occupancy
entry evidence
exit evidence
```

## 11.8 Step 6 — Demo 간 sequence alignment

각 demo의 segment 수가 다를 수 있으므로 normalized time만으로 경계를 평균내지 않는다.

segment descriptor distance를 사용한 dynamic programming sequence alignment를 수행한다.

허용 연산:

```text
match
merge two adjacent local segments
skip low-confidence local segment
```

금지:

```text
canonical phase 순서 역전
high-confidence event segment 무조건 삭제
```

alignment cost 예시:

```text
cost(i, j) =
    descriptor_distance(local_i, canonical_j)
  + duration_mismatch_penalty
  + transition_order_penalty
```

## 11.9 Step 7 — Canonical phase sequence 추론

초기 canonical sequence는 confidence가 가장 높은 medoid demo의 segment sequence로 시작한다.

그 후 반복한다.

```text
1. 모든 demo를 canonical sequence에 align
2. 각 canonical phase descriptor center 갱신
3. 지지 demo 비율이 낮은 phase 제거
4. 서로 매우 유사한 인접 phase merge
5. objective가 수렴할 때까지 반복
```

canonical phase 이름은 의미 label이 아닌 다음 형식을 사용한다.

```text
z0, z1, z2, ...
```

`Approach`, `Grasp`, `Transport`, `Place`는 시각적 해석 단계에서만 사람이 붙일 수 있으며 알고리즘 입력으로 사용하지 않는다.

## 11.10 Step 8 — Boundary merge 및 unstable boundary 제거

다음 boundary는 제거 또는 merge 대상이다.

```text
bootstrap support가 낮음
leave-one-feature-out에서 크게 이동
여러 demo 중 일부에만 존재
인접 phase descriptor가 사실상 동일
minimum duration 위반
```

## 11.11 Step 9 — 안정성 검증

### A. Bootstrap demo stability

training demo를 복원추출하여 segmentation 전체를 반복한다.

각 boundary의 canonical support를 계산한다.

### B. Temporal jitter stability

feature timestamp에 작은 jitter를 주고 boundary 이동량을 측정한다.

### C. Leave-one-feature-out stability

structural feature를 하나씩 제거하고 segmentation을 다시 수행한다.

한 feature 제거로 phase 수나 boundary가 크게 바뀌면 해당 결과는 취약하다.

### D. Filter sensitivity

허용된 smoothing window 범위에서 결과가 유지되는지 확인한다.

### E. Held-out demo sequence consistency

held-out demo를 canonical model에 align했을 때 비정상적인 insertion/deletion이 과도하지 않아야 한다.

## 11.12 최종 confidence

```text
segmentation_confidence =
    boundary_bootstrap_support
  * sequence_alignment_consistency
  * feature_ablation_stability
  * heldout_sequence_consistency
```

## 11.13 시각화

각 demo에 대해 다음 그림을 저장한다.

```text
subplot 1: normalized structural features
subplot 2: per-feature boundary evidence
subplot 3: combined evidence
subplot 4: selected boundaries and phase ids
subplot 5: canonical phase alignment
```

## 11.14 산출물

```text
artifacts/stage_06/segmentations/*.npz
artifacts/stage_06/canonical_phase_model.json
artifacts/stage_06/stability_report.json
artifacts/stage_06/plots/*.png
```

## 11.15 Gate 6 — Automatic Segmentation

모든 조건을 만족해야 PASS다.

```text
1. phase count가 config max를 넘지 않음
2. minimum phase duration 위반 없음
3. median bootstrap boundary support >= threshold
4. leave-one-feature-out median boundary shift <= threshold
5. held-out demo alignment cost <= threshold
6. pure pause 또는 짧은 detour가 독립 phase로 과도하게 생성되지 않음
7. 사람이 feature 이름을 바꿔도 동일 결과
8. reference visualization을 사람이 확인하고 물리적으로 불가능한 경계가 없음을 승인
```

여기서 사람의 확인은 phase label을 입력하는 것이 아니라 **출력 검증**이다.

사람이 결과를 수정해 학습 label로 다시 넣지 않는다.

---

# 12. Stage 7 — Phase Subgoal Inference

## 12.1 목적

canonical phase마다 어떤 feature가 변화해야 하고, 어떤 feature가 유지되어야 하는지를 demonstration에서 자동 추론한다.

## 12.2 Change feature 판정

phase `z_k`에서 feature `f`가 Change가 되려면 다음을 만족해야 한다.

```text
1. |median start-end delta|가 충분히 큼
2. demo 간 delta sign이 일치
3. phase 내부 slope가 일관됨
4. phase endpoint에서 변화가 대체로 완료됨
5. bootstrap confidence가 높음
```

`demo_direction`은 성공 demonstration에서 관측된 median delta의 부호다.

이것은 `higher_is_better`라는 전역 의미가 아니다.

예:

```text
Approach에 해당하는 어떤 latent phase에서
EEF-object distance가 일관되게 감소했다면
그 phase 안에서의 demonstrated direction은 감소다.
```

## 12.3 Hold feature 판정

다음 조건을 만족하면 Hold 후보가 된다.

```text
1. phase 후반 또는 endpoint에서 variance가 낮음
2. 여러 demo에서 reference center가 일치
3. phase 동안 해당 상태가 유지됨
4. reference distribution에서 벗어나면 segment descriptor가 크게 변함
```

## 12.4 Passive feature

Change 또는 Hold confidence가 낮은 feature는 Passive로 둔다.

Passive feature는 삭제하지 않는다.

FCM consequence target과 추후 reward 입력에는 유지할 수 있다.

## 12.5 Phase completion score

각 phase에 대해 structural feature만 사용하는 completion score를 정의한다.

이 score는 reward가 아니며, 해당 phase의 demonstrated subgoal 달성도를 측정하는 진단 함수다.

### Change feature contribution

feature `f`의 phase 시작값을 `x_start`, 현재값을 `x_t`, demonstrated endpoint를 `x_end`라고 할 때:

```text
progress_f(t) =
    clip(
        signed_projection(x_t - x_start, x_end - x_start)
        / max(|x_end - x_start|, epsilon),
        lower,
        upper
    )
```

### Hold feature contribution

```text
hold_f(t) = exp(- robust_distance(x_t, reference_distribution_f))
```

### Total completion score

```text
C_k(t) = weighted_mean(change_progress, hold_score)
```

weights는 subgoal inference confidence에서 가져온다.

## 12.6 Degradation target score

phase가 끝난 뒤 얼마나 subgoal이 열화되었는지를 측정한다.

```text
D_k = C_k(original endpoint) - C_k(perturbed endpoint)
```

따라서:

```text
D_k > 0  : perturbed trajectory가 phase subgoal을 열화
D_k ≈ 0  : 영향 없음
D_k < 0  : 오히려 subgoal completion이 개선됨
```

이 score는 structural subgoal만 평가한다.

energy, jerk 같은 passive quality feature의 좋고 나쁨을 여기서 하드코딩하지 않는다.

## 12.7 산출물

```text
artifacts/stage_07/subgoals.json
artifacts/stage_07/phase_completion_diagnostics.npz
artifacts/stage_07/subgoal_report.json
```

## 12.8 Gate 7

```text
PASS 조건
- 각 canonical phase에 최소 하나 이상의 high-confidence Change 또는 Hold feature 존재
- demonstrated phase endpoint의 completion score가 phase start보다 높음
- held-out demo에서도 같은 phase completion trend가 유지
- random time-shuffled phase에서는 confidence가 감소
```

Stage 7까지가 Milestone A다.

---

# 13. Stage 8 — Perturbation Rollout Dataset

## 13.1 목적

동일한 simulator state에서 demo action과 perturbed action을 분기 실행하여 perturbation의 feature consequence를 측정한다.

## 13.2 핵심 실험 단위

각 sample은 다음 방식으로 만든다.

```text
1. demo timestep t의 exact simulator state 복원
2. Branch A: 원래 demo action sequence 실행
3. Branch B: 지정 window 동안 perturbation을 더한 action 실행
4. window 이후에는 원래 demo action sequence로 복귀
5. 여러 horizon에서 feature 차이 측정
```

이 구조는 초기 state와 future demo action을 최대한 동일하게 유지하여 perturbation의 인과 효과를 분리한다.

## 13.3 Perturbation 식

```text
a_pert(t+j) = clip(
    a_demo(t+j) + lambda * mask(j) * d,
    action_low,
    action_high
)
```

여기서:

```text
d       : action_dim unit direction
lambda  : magnitude
mask(j) : perturbation window envelope
```

초기 구현 envelope:

```text
rectangular
```

선택적 확장:

```text
triangular
smooth cosine
```

## 13.4 후보 action subspace

OSC_POSE 기준:

```text
position : indices [0, 1, 2]
rotation : indices [3, 4, 5]
gripper  : index   [6]
mixed    : selected combinations
```

실제 index는 adapter의 `ActionLayout`에서 읽는다.

## 13.5 Sampling 전략

모든 phase와 progress 구간이 균형 있게 포함되도록 stratified sampling한다.

```text
phase id
phase progress bin
perturbation subspace
magnitude bin
window duration bin
action direction family
```

phase progress 기본 bin:

```text
[0.0, 0.25)
[0.25, 0.50)
[0.50, 0.75)
[0.75, 1.0]
```

## 13.6 Direction sampling

초기 dataset에는 다음을 혼합한다.

```text
1. positive/negative coordinate basis
2. random unit directions within each subspace
3. low-discrepancy sphere directions
4. demo-action parallel / anti-parallel directions
```

특정 subgoal 방향을 미리 action space에 역매핑하지 않는다.

FCM이 consequence를 학습할 수 있도록 충분히 다양한 perturbation을 수집한다.

## 13.7 Magnitude sampling

각 action dimension의 유효 범위를 정규화한 뒤 작은 perturbation부터 시작한다.

```text
lambda_normalized ∈ {0.0, 0.05, 0.10, 0.20, 0.30}
```

실제 값은 action range에 곱한다.

`lambda=0` sample을 전체 dataset의 일정 비율 포함한다.

권장:

```text
zero perturbation samples = 10%
```

## 13.8 Residual target

horizon `h`에서:

```text
r_(t,h) = phi_pert(t+h) - phi_demo(t+h)
```

여러 horizon을 동시에 저장한다.

권장:

```text
horizons_sec = [0.05, 0.10, 0.20, 0.40, phase_end]
```

## 13.9 Feasibility flags

각 rollout에서 저장:

```text
action clipping occurred
joint limit proximity
workspace violation
object dropped
numerical instability
task success
phase completion score
```

## 13.10 산출물

```text
artifacts/stage_08/fcm_dataset.hdf5
artifacts/stage_08/dataset_manifest.json
artifacts/stage_08/sampling_balance.csv
```

## 13.11 Gate 8 — Zero Perturbation

```text
lambda = 0
→ residual target ≈ 0
```

PASS 조건:

```text
mean absolute residual <= tolerance
95th percentile residual <= tolerance
```

실패하면 state branching 또는 replay가 잘못된 것이므로 FCM 학습을 금지한다.

---

# 14. Stage 9 — Residual Forward Consequence Model

## 14.1 목적

현재 context와 action perturbation을 입력받아 future feature residual을 예측한다.

## 14.2 입력

```text
normalized feature_t
normalized demo action_t
normalized perturbation delta
canonical phase embedding
phase progress
perturbation window duration
goal context embedding
horizon embedding
```

## 14.3 출력

```text
predicted feature residual for each horizon
```

선택적 auxiliary output:

```text
feasibility probability
action clipping probability
object drop probability
```

단 auxiliary output이 final simulator validation을 대체하지 않는다.

## 14.4 모델 구조

초기 고정안:

```text
ensemble of MLP residual predictors
```

권장 기본값:

```text
ensemble members : 5
hidden layers    : [256, 256, 128]
activation       : SiLU or ReLU
normalization    : LayerNorm optional
output           : H × F residual
loss             : weighted Huber loss
```

sequence model은 Stage 9 baseline이 통과한 뒤에만 비교한다.

## 14.5 Loss

```text
L_residual = weighted_mean_f,h(
    Huber(r_pred - r_true)
)
```

feature weight는 scale 차이를 보정하기 위한 수치적 weight만 허용한다.

사람이 quality 의미를 이용해 weight를 지정하지 않는다.

선택적 auxiliary loss:

```text
L_total = L_residual + alpha * L_feasibility
```

## 14.6 Ensemble uncertainty

각 member prediction의 표준편차를 epistemic uncertainty proxy로 사용한다.

```text
mean prediction = mean(member predictions)
uncertainty     = std(member predictions)
```

## 14.7 데이터 분할

frame 단위 random split을 금지한다.

반드시 demonstration 단위로 분할한다.

```text
train demos      70%
validation demos 15%
test demos       15%
```

demo 수가 적으면 group K-fold를 사용한다.

## 14.8 평가 지표

### Prediction metrics

```text
feature-wise MAE
feature-wise normalized MAE
feature-wise R²
phase-wise R²
horizon-wise error
```

### Screening metrics

FCM의 핵심 지표다.

```text
Spearman correlation between predicted and true degradation score
Top-K recall
NDCG@K
uncertainty calibration
random-screening 대비 simulator hit-rate 향상
```

## 14.9 Top-K recall 정의

실제 simulator 기준 상위 M개의 degradation candidate 중 FCM이 고른 Top-K에 포함된 비율을 측정한다.

```text
TopKRecall = |TrueTopM ∩ PredictedTopK| / M
```

## 14.10 산출물

```text
artifacts/stage_09/fcm_ensemble.pt
artifacts/stage_09/fcm_scalers.json
artifacts/stage_09/fcm_metrics.json
artifacts/stage_09/ranking_report.json
```

## 14.11 Gate 9 — FCM Screening

```text
PASS 조건
- held-out demo Top-K recall이 random baseline보다 유의하게 높음
- severe uncertainty 구간이 오류 증가와 양의 상관
- zero perturbation prediction이 0 근처
- phase별 최소 성능 기준 충족
```

단순 평균 MAE가 낮아도 Top-K recall이 낮으면 FAIL이다.

Stage 9까지가 Milestone B다.

---

# 15. Stage 10 — Degradation Hypothesis Generation

## 15.1 목적

PhaseSubgoal에서 어떤 종류의 열화를 만들 것인지 자동 생성한다.

## 15.2 Change opposition hypothesis

Change feature의 demonstrated direction을 방해하는 가설이다.

예:

```text
demo에서 feature가 감소했다
→ perturbed endpoint에서 감소량을 줄이거나 반대로 증가시키는 consequence를 목표
```

정의:

```text
target = reduce demonstrated signed progress
```

여기서 action direction을 직접 정하지 않는다.

feature consequence 수준의 가설만 만든다.

## 15.3 Hold violation hypothesis

Hold feature가 reference distribution을 유지하지 못하게 만드는 가설이다.

정의:

```text
target = increase robust distance from phase hold distribution
```

## 15.4 Composite hypothesis

한 phase에서 여러 structural feature가 공동으로 subgoal을 정의하면 weighted composite를 생성할 수 있다.

```text
composite degradation score =
    weighted sum of Change progress loss
  + weighted sum of Hold distribution violation
```

weights는 Stage 7 confidence에서 가져온다.

## 15.5 Passive feature 처리

Passive feature만으로 hypothesis를 만들지 않는다.

다만 모든 candidate rollout에서 passive feature consequence를 저장한다.

## 15.6 Hypothesis pruning

다음은 제거한다.

```text
subgoal confidence가 낮음
held-out demo에서 방향이 불일치
endpoint reference가 불안정
서로 사실상 중복된 hypothesis
```

## 15.7 산출물

```text
artifacts/stage_10/degradation_hypotheses.json
```

## 15.8 Gate 10

```text
PASS 조건
- 각 hypothesis가 특정 canonical phase와 연결됨
- target score가 수식으로 재현 가능
- feature 이름의 의미를 사용하지 않음
- low-confidence subgoal에서 hypothesis를 생성하지 않음
```

---

# 16. Stage 11 — Candidate Generation and FCM Screening

## 16.1 목적

각 degradation hypothesis를 유발할 가능성이 있는 action perturbation 후보를 생성하고 FCM으로 Top-K를 고른다.

## 16.2 Candidate parameterization

한 candidate는 다음으로 정의한다.

```text
canonical phase id
start fraction within phase
duration fraction within phase
action subspace
unit direction vector d
```

`lambda`는 candidate identity에 포함하지 않고 Stage 12에서 별도로 calibration한다.

## 16.3 초기 fixed-direction 방식

MVP에서는 perturbation window 동안 같은 direction `d`를 적용한다.

```text
d_t = d for all t in perturbation window
```

이 방식의 장점:

```text
구현 단순
해석 가능
lambda calibration 용이
candidate family 정의 명확
```

한계:

```text
상태 의존 열화에는 부족할 수 있음
grasp처럼 민감한 구간에서 valid lambda 범위가 매우 작을 수 있음
```

해결 정책:

```text
1차 구현에서는 fixed direction 유지
valid lambda 범위가 없는 candidate는 reject
추후 state-dependent perturbation은 별도 ablation으로 추가
```

## 16.4 후보 direction 생성

각 action subspace에서 다음을 생성한다.

```text
coordinate basis ±e_i
random unit vectors
low-discrepancy unit vectors
current demo action parallel/anti-parallel
```

후보 수 예시:

```text
position directions : 32
rotation directions : 32
gripper directions  : 2
mixed directions    : 32
start fractions     : [0.0, 0.25, 0.50, 0.75]
duration fractions  : [0.10, 0.25, 0.50, 1.00]
```

실제 수는 simulator budget에 맞춰 config에서 조절한다.

## 16.5 FCM consequence prediction

각 candidate에 대해 작은 reference magnitude `lambda_probe`를 사용하여 residual을 예측한다.

```text
predicted perturbed feature
= demo future feature + predicted residual
```

그 결과 Stage 7 completion score를 재계산한다.

## 16.6 Predicted degradation score

```text
D_pred = C_original - C_predicted_perturbed
```

큰 값일수록 target subgoal을 더 열화시킬 것으로 예상한다.

## 16.7 Uncertainty-aware screening

단순히 `D_pred`가 큰 후보만 고르지 않는다.

```text
screening_score =
    D_pred
  - uncertainty_penalty * U_pred
  - infeasibility_penalty
```

## 16.8 Diversity selection

Top-K가 거의 동일한 direction으로 채워지지 않도록 greedy diversity selection을 사용한다.

candidate distance는 다음을 반영한다.

```text
action direction cosine distance
start fraction difference
duration difference
action subspace difference
```

## 16.9 Phase별 budget

각 phase와 hypothesis마다 최소 K개를 simulator로 보낸다.

특정 phase가 높은 score를 독점하지 못하게 한다.

## 16.10 산출물

```text
artifacts/stage_11/all_candidates.json
artifacts/stage_11/screened_candidates.json
artifacts/stage_11/screening_diagnostics.csv
```

## 16.11 Gate 11

offline benchmark candidate subset에서 다음을 확인한다.

```text
FCM Top-K simulator hit rate > random Top-K
uncertainty penalty 적용이 catastrophic candidate 비율을 감소
candidate diversity 기준 충족
```

---

# 17. Stage 12 — Exact Simulator Validation

## 17.1 목적

FCM이 선별한 candidate를 실제 simulator에서 실행하고, 성공을 유지하면서 단계적으로 열화되는 family만 채택한다.

## 17.2 동일 초기 조건

candidate 비교 시 다음을 동일하게 고정한다.

```text
same demonstration
same phase instance
same start simulator state
same future demo action sequence
same perturbation direction/window
only lambda changes
```

## 17.3 Lambda search의 정의

각 candidate마다 독립적으로 `lambda_max`를 찾는다.

`lambda_max`는 다음을 만족하는 최대 magnitude다.

```text
1. task success 유지
2. simulator numerical stability 유지
3. forbidden workspace violation 없음
4. excessive action clipping 없음
5. target degradation score가 최소 효과 크기 이상
```

## 17.4 Lambda search 알고리즘

### Step A — Bracketing

```text
lambda_low = 0
lambda_high = initial_probe
```

`lambda_high`를 배수로 증가시키며 다음 중 하나가 발생할 때까지 반복한다.

```text
task failure
feasibility violation
max allowed lambda 도달
```

### Step B — Binary search

마지막 valid lambda와 첫 invalid lambda 사이에서 binary search한다.

```text
while interval > tolerance:
    lambda_mid = (valid + invalid) / 2
    rollout(lambda_mid)
    if valid:
        valid = lambda_mid
    else:
        invalid = lambda_mid
```

최종 valid 값을 `lambda_feasible_max`로 둔다.

### Step C — Effect threshold 확인

`lambda_feasible_max`에서 target degradation이 충분하지 않으면 candidate를 reject한다.

```text
D(lambda_feasible_max) >= min_effect_size
```

## 17.5 Nested levels

기본 level:

```text
Original : 0.00 * lambda_max
Mild     : 0.25 * lambda_max
Medium   : 0.50 * lambda_max
Severe   : 0.75 * lambda_max
Max      : 1.00 * lambda_max
```

Preference 단계에서 4단계만 필요하면 Max를 diagnostic으로만 유지할 수 있다.

## 17.6 Validation metrics

각 level에서 다음을 저장한다.

```text
task success
phase completion score
target degradation score
all raw and normalized feature trajectories
path length
energy proxy
jerk
action magnitude
object slip
contact stability
action clipping ratio
recovery after perturbation
final goal error
```

quality metric의 방향은 여기서 ranking label로 하드코딩하지 않는다.

## 17.7 Monotonicity

이상적인 조건:

```text
D(Original) < D(Mild) < D(Medium) < D(Severe) <= D(Max)
```

noise tolerance를 허용한 monotonicity score를 사용한다.

예:

```text
pairwise ordered fraction
Spearman correlation(lambda, degradation score)
number of violations
```

## 17.8 Gradedness

단순히 순서만 맞고 차이가 거의 0인 family는 유효하지 않다.

다음 효과 크기를 확인한다.

```text
D(Mild)   - D(Original)
D(Medium) - D(Mild)
D(Severe) - D(Medium)
```

최소 두 단계 이상에서 `min_step_effect`를 넘어야 한다.

## 17.9 Success preservation

기본 채택 조건:

```text
Original, Mild, Medium, Severe 모두 task success = True
```

Max에서만 failure가 발생하면 Max는 family에서 제외하고 Severe까지만 사용할 수 있다.

Mild부터 failure면 candidate를 reject한다.

## 17.10 Recovery property

perturbation window가 끝난 뒤 demo action sequence로 복귀했을 때 task가 회복되는지를 측정한다.

좋은 structured degradation은 다음 특성을 가진다.

```text
perturbation 동안 subgoal 진행 저하
이후 demo action으로 일정 부분 회복
최종 task success 유지
전체 trajectory는 원본보다 비효율적 또는 불안정
```

## 17.11 Family 채택 조건

모든 조건을 만족해야 한다.

```text
1. target degradation effect >= threshold
2. monotonicity score >= threshold
3. gradedness score >= threshold
4. Mild/Medium/Severe task success 유지
5. clipping ratio <= threshold
6. no numerical instability
7. 동일 candidate를 다른 demo에서도 재검증했을 때 일정 비율 이상 성공
```

## 17.12 Cross-demo validation

한 demo에서만 작동하는 candidate를 일반적인 degradation mechanism으로 간주하지 않는다.

동일 canonical phase의 다른 demo instance에 candidate parameter를 적용한다.

```text
same action direction
same relative start/duration
candidate-specific lambda 재calibration 가능
```

family confidence에는 cross-demo transfer rate를 포함한다.

## 17.13 산출물

```text
artifacts/stage_12/validated_families.json
artifacts/stage_12/validated_trajectories.hdf5
artifacts/stage_12/lambda_search_logs.json
artifacts/stage_12/family_metrics.csv
artifacts/stage_12/plots/*.png
```

## 17.14 Gate 12 — Simulator-Validated Degradation

```text
PASS 조건
- 최소 family 수 충족
- phase별 최소 한 개 이상의 validated family 존재 또는 실패 이유 보고
- nested degradation monotonicity 충족
- task success preservation 충족
- FCM screening이 random screening보다 simulator 효율 향상
- held-out demo transfer rate 충족
```

Stage 12까지가 Milestone C이며 현재 확정 범위의 완료 지점이다.

---

# 18. Degradation 로직의 논리적 설명

## 18.1 왜 phase segmentation이 먼저 필요한가

같은 action perturbation도 task의 어느 시점에 적용하느냐에 따라 결과가 다르다.

```text
접근 중 위치 perturbation
→ object 접근 실패 또는 우회

grasp 중 gripper perturbation
→ 접촉 불안정

transport 중 rotation perturbation
→ object slip 또는 재정렬

place 중 위치 perturbation
→ 최종 오차 증가
```

따라서 전체 trajectory에 무작위 noise를 넣는 것보다 phase와 subgoal을 먼저 추론하고, 해당 phase의 subgoal을 방해하는 perturbation을 찾는 것이 구조적으로 해석 가능하다.

## 18.2 왜 subgoal을 feature 변화로 정의하는가

phase 이름 자체는 action consequence를 수치화하지 못한다.

예를 들어 `Approach`라는 이름보다 다음 표현이 계산 가능하다.

```text
EEF-object distance가 감소한다.
gripper 상태는 아직 크게 변하지 않는다.
object position은 거의 유지된다.
```

이처럼 phase subgoal을 Change/Hold feature의 분포와 변화 방향으로 표현하면, perturbation 전후의 subgoal completion 차이를 직접 계산할 수 있다.

## 18.3 왜 FCM이 필요한가

7차원 action space에서 direction, start time, duration을 조합하면 candidate 수가 빠르게 증가한다.

모든 candidate를 simulator에서 여러 lambda로 실행하는 것은 비싸다.

FCM은 다음 질문에 빠르게 답한다.

> 이 perturbation을 적용하면 phase endpoint의 feature가 어느 방향으로 얼마나 변할 가능성이 있는가?

FCM은 높은 degradation을 예상하는 후보를 선별하지만, 모델 오차가 있으므로 최종 판정은 simulator가 수행한다.

## 18.4 왜 residual FCM인가

절대 미래 feature를 예측하면 demo dynamics 전체를 다시 학습해야 한다.

Residual FCM은 다음 차이만 예측한다.

```text
perturbed future feature - original demo future feature
```

동일한 초기 state와 future action sequence를 기준으로 하기 때문에 perturbation의 영향에 집중할 수 있다.

## 18.5 왜 lambda를 candidate마다 따로 찾는가

같은 magnitude라도 action direction과 phase에 따라 영향이 다르다.

```text
position x 방향 0.1
rotation yaw 방향 0.1
gripper 방향 0.1
```

은 물리적 의미와 민감도가 완전히 다르다.

따라서 전역 `lambda_max`를 쓰지 않고 candidate마다 성공이 깨지기 직전의 최대 유효 magnitude를 찾는다.

## 18.6 왜 simulator validation이 필요한가

FCM은 다음을 완벽히 예측하지 못할 수 있다.

```text
contact discontinuity
object drop
joint limit
controller clipping
nonlinear recovery
long-horizon compounding error
```

따라서 FCM은 비용 절감용 screening이며, 실제 degradation hierarchy는 simulator rollout으로만 확정한다.

## 18.7 왜 성공을 유지하는 열화를 선호하는가

단순 실패는 쉽게 만들 수 있지만 유용한 preference를 제공하지 못할 수 있다.

```text
성공 vs 실패
```

만 반복하면 reward가 task success만 학습하고 세밀한 efficiency 차이를 학습하지 못할 가능성이 있다.

따라서 본 파이프라인은 가능한 한 다음 family를 목표로 한다.

```text
모두 성공하지만
경로, 안정성, phase completion, recovery가 점진적으로 나빠지는 trajectory
```

## 18.8 현재 fixed된 degradation의 의미

현재 확정된 것은 다음이다.

```text
1. PhaseSubgoal이 feature-space 열화 목표를 정의한다.
2. Candidate generator가 action-space 후보를 생성한다.
3. FCM이 feature consequence를 예측해 Top-K를 고른다.
4. Simulator가 candidate별 lambda_max를 찾는다.
5. 성공을 유지하면서 monotonic한 family만 채택한다.
```

아직 확정하지 않은 것은 다음이다.

```text
Validated family로 어떤 reward를 학습할지
Original보다 좋은 policy를 어떤 방식으로 탐색할지
reward extrapolation을 어떻게 제어할지
```

---

# 19. Stage 13 이후 — Provisional BfD Interface

이 절은 구현 확정안이 아니라 Stage 12 artifact와 후속 연구 사이의 계약이다.

## 19.1 Stage 12가 제공하는 것

```text
phase-conditioned successful degradation families
Original > Mild > Medium > Severe를 지지하는 simulator evidence
all structural and passive feature trajectories
candidate provenance
lambda calibration logs
```

## 19.2 아직 해결해야 하는 핵심 문제

Degradation family는 `Original보다 나쁜 방향`을 제공한다.

그러나 Better-than-Demonstrator를 주장하려면 다음이 추가로 필요하다.

```text
Original보다 좋은 영역을 어떻게 탐색하는가?
reward가 degradation 반대 방향으로 외삽해도 되는가?
외삽 reward exploitation을 어떻게 검증하는가?
개선 candidate를 simulator에서 직접 검증할 것인가?
```

## 19.3 검토 가능한 후속 방향

### Option A — Structured D-REX style reward extrapolation

```text
validated degradation ranking
→ preference reward
→ conservative policy improvement
```

필요한 안전장치:

```text
behavior constraint
reward ensemble uncertainty
held-out family preference accuracy
simulator success guard
reward exploitation detection
```

### Option B — Improvement-side counterfactual validation

```text
degradation 반대 consequence를 예측
→ improvement candidate 생성
→ simulator에서 Original보다 실제로 나은지 검증
→ Improved > Original anchor 추가
```

이 방향은 외삽 의존성을 줄일 수 있으나 구현 난이도가 증가한다.

### Option C — Direct constrained policy improvement

학습 reward 전체를 믿기보다 success constraint 아래에서 명시적 efficiency metric 또는 validated local direction을 사용하는 방식이다.

## 19.4 현재 코드 정책

Stage 12 이전 코드에는 특정 BfD 옵션을 강제로 넣지 않는다.

`provisional_bfd/`는 별도 모듈로 유지하여 segmentation, FCM, degradation core와 분리한다.

---

# 20. 전체 Gate 요약

| Gate | 이름 | 핵심 조건 |
|---|---|---|
| 1 | Demo Validity | 성공 demo 수, metadata, action layout |
| 2 | Demo I/O | shape, checksum, deterministic load |
| 3 | State Restore | exact simulator state 복원 |
| 4 | Feature Engine | feature order, valid mask, scaler |
| 5 | Feature Profiling | synthetic discrimination, bootstrap confidence |
| 6 | Segmentation | multi-demo consistency, stability, no phase explosion |
| 7 | Subgoal Inference | change/hold confidence, held-out consistency |
| 8 | Zero Perturbation | delta=0 residual≈0 |
| 9 | FCM Screening | Top-K recall > random |
| 10 | Hypothesis Validity | subgoal-derived, reproducible target |
| 11 | Candidate Screening | uncertainty/diversity, simulator hit-rate |
| 12 | Simulator Validation | success-preserving monotonic degradation |

---

# 21. 기본 Configuration 초안

아래 값은 시작점이며 validation 결과에 따라 train split에서만 조정한다.

```yaml
project:
  seed: 42
  target_task: PickPlaceSingleObject
  controller: OSC_POSE

features:
  robust_scaler:
    center: median
    scale: iqr
    epsilon: 1.0e-6
    clip: 10.0
  smoothing:
    enabled: true
    method: savgol
    window_sec: 0.15
    polyorder: 3

profiling:
  bootstrap_samples: 100
  structural_score_threshold: 0.60
  confidence_threshold: 0.70
  noise_robustness_threshold: 0.60

segmentation:
  multiscale_windows_sec: [0.10, 0.20, 0.40, 0.80]
  min_phase_duration_sec: 0.35
  max_phase_count: 10
  edge_margin_sec: 0.15
  complexity_penalty: 1.0
  bootstrap_samples: 100
  min_boundary_support: 0.70
  max_leave_one_feature_shift_sec: 0.20
  require_human_output_inspection: true

subgoal:
  min_change_effect: 0.50
  min_direction_consistency: 0.75
  min_hold_consistency: 0.75
  epsilon: 1.0e-6

fcm_dataset:
  zero_perturbation_fraction: 0.10
  phase_progress_bins: [0.0, 0.25, 0.50, 0.75, 1.0]
  lambda_normalized: [0.0, 0.05, 0.10, 0.20, 0.30]
  horizons_sec: [0.05, 0.10, 0.20, 0.40]
  include_phase_end_horizon: true

fcm:
  ensemble_size: 5
  hidden_dims: [256, 256, 128]
  activation: silu
  loss: huber
  batch_size: 512
  learning_rate: 3.0e-4
  early_stopping_patience: 20
  top_k: 20

candidate_search:
  coordinate_basis: true
  random_directions_per_subspace: 32
  start_fractions: [0.0, 0.25, 0.50, 0.75]
  duration_fractions: [0.10, 0.25, 0.50, 1.0]
  lambda_probe: 0.05
  uncertainty_penalty: 1.0
  diversity_weight: 0.25
  top_k_per_hypothesis: 10

sim_validation:
  initial_lambda_probe: 0.05
  lambda_growth_factor: 2.0
  lambda_search_tolerance: 0.01
  lambda_level_fractions: [0.0, 0.25, 0.50, 0.75, 1.0]
  max_action_clipping_ratio: 0.05
  min_effect_size: 0.10
  min_step_effect: 0.03
  min_monotonicity_score: 0.80
  require_success_levels: [0.0, 0.25, 0.50, 0.75]
```

---

# 22. 필수 테스트 목록

## 22.1 Unit tests

```text
test_demo_loader_shapes
test_state_restore_roundtrip
test_feature_registry_order
test_robust_scaler_no_leakage
test_event_score_binary_transition
test_trend_score_monotonic_signal
test_noise_not_structural
test_segmentation_min_duration
test_segmentation_no_name_dependency
test_subgoal_change_direction
test_subgoal_hold_distribution
test_zero_perturbation_residual
test_fcm_output_shape
test_candidate_unit_direction
test_lambda_binary_search
test_family_monotonicity_metric
```

## 22.2 Synthetic integration tests

```text
Synthetic Pick-and-Place-like feature trajectory
- phase A: distance decreases
- phase B: contact switches on
- phase C: object height increases
- phase D: object-goal distance decreases
- phase E: height decreases and goal contact stabilizes
```

여기에 다음 노이즈를 추가한다.

```text
pause
detour
single-frame spike
temporal stretching
measurement noise
missing frames
```

알고리즘은 정확한 frame 일치보다 canonical sequence와 근사 boundary를 복원해야 한다.

## 22.3 Regression tests

reference demos에 대해 다음을 version control한다.

```text
feature order
canonical phase count
boundary confidence range
FCM Top-K recall range
validated family count range
```

정확한 boundary index를 영구 하드코딩하지 않고 허용 구간을 사용한다.

---

# 23. Claude Code 구현 순서

Coding agent는 전체를 한 번에 구현하지 않는다.

## Milestone A1 — Data foundation

```text
Stage 1
Stage 2
Stage 3
Stage 4
```

완료 조건:

```text
demo.hdf5
→ EpisodeData
→ exact replay
→ RawTrajectory
→ FeatureTrajectory
```

## Milestone A2 — Automatic segmentation

```text
Stage 5
Stage 6
Stage 7
```

완료 조건:

```text
FeatureProfileSet
PhaseSegmentationSet
CanonicalPhaseModel
PhaseSubgoalSet
```

## Milestone B — FCM

```text
Stage 8
Stage 9
```

완료 조건:

```text
zero perturbation gate 통과
held-out Top-K recall > random
```

## Milestone C — Degradation

```text
Stage 10
Stage 11
Stage 12
```

완료 조건:

```text
validated success-preserving degradation families 생성
```

## 구현 운영 규칙

각 Stage마다 coding agent가 다음 순서를 따른다.

```text
1. stage 문서 읽기
2. 구현 계획 작성
3. public API와 dataclass 먼저 작성
4. unit test 작성
5. 최소 구현
6. synthetic test
7. reference demo integration test
8. artifact 생성
9. 시각화 확인
10. Gate PASS/FAIL 기록
11. git diff 검토
12. commit
```

---

# 24. Stage 완료 상태 표기 규칙

`docs/IMPLEMENTATION_STATUS.md`에서 다음 상태만 사용한다.

```text
NOT_STARTED
INTERFACE_DEFINED
IMPLEMENTED_UNVERIFIED
GATE_FAILED
GATE_PASSED
```

예시:

| Stage | Status | Last artifact | Gate |
|---|---|---|---|
| 4 Feature Engine | GATE_PASSED | features_v3.npz | PASS |
| 5 Feature Profiling | IMPLEMENTED_UNVERIFIED | profiles_debug.json | NOT RUN |
| 9 Residual FCM | NOT_STARTED | — | — |

`설계 완료`와 `코드 완료`를 같은 의미로 사용하지 않는다.

---

# 25. 연구적으로 중요한 Ablation

Stage 12까지의 핵심 novelty를 검증하기 위해 다음 ablation을 필수로 둔다.

```text
A. random Gaussian degradation
   vs phase-conditioned structured degradation

B. no segmentation
   vs automatic phase segmentation

C. single-feature boundary
   vs multivariate robust segmentation

D. no cross-demo alignment
   vs canonical phase alignment

E. no stability filtering
   vs bootstrap + leave-one-feature-out filtering

F. no FCM screening
   vs FCM screening

G. deterministic single FCM
   vs ensemble uncertainty FCM

H. global lambda
   vs candidate-specific lambda calibration

I. allow failure negatives
   vs success-preserving degradation only
```

평가 지표:

```text
segmentation stability
canonical sequence consistency
simulator calls per validated family
FCM Top-K recall
validated family yield
monotonicity
success preservation
cross-demo transfer rate
```

---

# 26. 최종 Acceptance Checklist

## Demonstration Understanding

```text
[ ] 10개 이상의 successful demo 로드
[ ] exact simulator restore gate 통과
[ ] feature leakage 없는 robust scaling
[ ] name-independent feature profiling
[ ] multi-demo automatic segmentation
[ ] bootstrap boundary support 보고
[ ] leave-one-feature-out stability 보고
[ ] held-out demo canonical alignment 통과
[ ] Change/Hold subgoal inference 통과
```

## FCM

```text
[ ] same-state branch rollout 구현
[ ] lambda=0 residual gate 통과
[ ] demo-level train/val/test split
[ ] ensemble residual predictor 학습
[ ] held-out Top-K recall > random
[ ] uncertainty calibration 보고
```

## Degradation

```text
[ ] subgoal-derived hypothesis 생성
[ ] phase-conditioned candidate 생성
[ ] FCM uncertainty-aware Top-K screening
[ ] candidate별 lambda bracketing + binary search
[ ] Mild/Medium/Severe success 유지
[ ] monotonic degradation family 생성
[ ] cross-demo transfer 검증
[ ] validated families artifact 저장
```

## BfD claim

```text
[ ] Stage 13 이후 알고리즘 별도 확정
[ ] Original보다 좋은 data 또는 정당한 extrapolation 논리 확보
[ ] reward exploitation guard 구현
[ ] improved policy simulator 검증
```

위 네 항목이 완료되기 전에는 Better-than-Demonstrator를 최종 성과로 주장하지 않는다.

---

# 27. 최종 파이프라인 한 줄 요약

```text
사람은 feature 계산 함수만 정의한다
→ 알고리즘이 여러 suboptimal demo의 feature dynamics를 분석한다
→ noise와 feature 제거에도 안정적인 phase sequence를 자동 추론한다
→ 각 phase의 Change/Hold subgoal을 feature-space로 정의한다
→ 동일 simulator state에서 action perturbation consequence dataset을 만든다
→ residual FCM이 많은 action 후보의 future feature consequence를 선별한다
→ simulator가 candidate별 lambda_max와 nested degradation을 직접 검증한다
→ 성공을 유지하면서 Original > Mild > Medium > Severe인 family만 저장한다
→ 그 이후 Better-than-Demonstrator 방법은 별도 연구 단계에서 확정한다
```

---

# 28. 현재 구현의 우선순위

가장 먼저 구현할 것은 FCM이나 reward network가 아니다.

```text
Priority 1
Demo Loader
→ State Replay
→ Feature Engine

Priority 2
Feature Profiler
→ Robust Multi-Demo Segmentation
→ Canonical Phase Alignment
→ Subgoal Inference

Priority 3
Perturbation Dataset
→ Residual FCM

Priority 4
Hypothesis
→ Candidate Search
→ Simulator Validation
```

특히 Stage 6의 stability report가 Gate를 통과하지 않으면 FCM과 degradation 구현을 진행하지 않는다.

잘못된 phase와 subgoal을 기반으로 만든 FCM dataset은 이후 모든 결과를 오염시키기 때문이다.
