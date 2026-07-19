# BfD 전체 파이프라인 구현 계획서 v5  
## 기존 5개 코드를 유지하면서 단계적으로 수정하는 방식

---

# 0. 수정 방향

본 프로젝트는 새로운 package 구조나 새로운 모듈 집합을 다시 만드는 방식으로 구현하지 않는다.

현재 존재하는 다음 다섯 개의 Python 파일을 **실행 가능한 baseline**으로 유지하고, 전체 연구 파이프라인의 데이터 흐름에 맞게 파일 내부를 단계적으로 수정한다.

```text
collect_demo.py
feature_select.py
phase_segment.py
fcm.py
degradation.py
```

핵심 원칙은 다음과 같다.

1. 이미 정상 동작하는 코드는 최대한 보존한다.
2. 각 파일의 현재 public function과 실행 흐름을 먼저 확인한다.
3. 필요한 기능을 같은 파일 안에 추가하거나 기존 함수를 수정한다.
4. 기존 코드를 한 번에 전부 폐기하거나 새 파일로 다시 작성하지 않는다.
5. 한 단계의 출력이 확인된 뒤 다음 파일을 수정한다.
6. 전체 실행 순서는 항상 동일하게 유지한다.

```text
collect_demo.py
→ feature_select.py
→ phase_segment.py
→ fcm.py
→ degradation.py
→ Validated Degradation Families
```

현재 확정 범위는 structured degradation 생성까지다.

```text
Preference Reward Learning
Policy Improvement
SAC / PPO
Better-than-Demonstrator Evaluation
```

은 후속 단계로 분리한다.

---

# 1. 현재 코드의 활용 원칙

## 1.1 새로 작성하지 않고 수정하는 이유

현재 코드에는 이미 다음과 같은 유용한 기능이 존재한다.

- robosuite interactive teleoperation
- OSC_POSE 및 OSC_POSITION controller 지원
- demonstration HDF5 저장
- feature 계산식
- robosuite environment replay helper
- object 및 contact signal 추출
- phase segmentation prototype
- action perturbation helper
- MLP 기반 FCM
- action subspace candidate 생성
- simulator trajectory execution
- degradation trajectory 저장 및 시각화

따라서 같은 기능을 새로운 package에서 다시 구현하면 다음 문제가 발생할 수 있다.

- 기존 코드와 새 코드의 HDF5 schema 불일치
- feature 계산식 중복
- object 및 contact resolver 중복
- action clipping 방식 불일치
- 기존에 해결된 robosuite version compatibility 문제 재발
- 어떤 구현이 기준인지 불명확해짐

본 계획은 기존 기능을 그대로 활용하면서, 연구 설계와 맞지 않는 부분만 교체한다.

---

## 1.2 파일별 처리 방식

| 파일 | 현재 코드에서 유지할 부분 | 수정할 핵심 부분 |
|---|---|---|
| `collect_demo.py` | teleoperation, controller 선택, HDF5 저장 | demo 검사, replay, raw trajectory 추출 추가 |
| `feature_select.py` | feature 계산식, action perturbation helper | 수동 KIND/BOUNDARY 의존 제거, automatic profiler 추가 |
| `phase_segment.py` | replay helper, object/contact resolver | multivariate segmentation, multi-demo alignment, subgoal inference |
| `fcm.py` | rollout 구조, MLP, action subspace 후보 | baseline-relative residual, phase input, group split, screening 평가 |
| `degradation.py` | action sequence 생성, simulator executor, trajectory 저장 | hypothesis 생성, FCM Top-K, candidate별 lambda, monotonic validation |

---

# 2. 최종 프로젝트 구조

Python source는 기존 다섯 개만 사용한다.

```text
ROBOT_BFD/
├── collect_demo.py
├── feature_select.py
├── phase_segment.py
├── fcm.py
├── degradation.py
│
├── data/
│   └── demos/
│       ├── demo_000/
│       │   └── demo.hdf5
│       └── ...
│
├── artifacts/
│   ├── raw/
│   ├── features/
│   ├── segmentation/
│   ├── fcm/
│   └── degradation/
│
├── plots/
│   ├── features/
│   ├── segmentation/
│   ├── fcm/
│   └── degradation/
│
└── README.md
```

다음 source file은 만들지 않는다.

```text
utils.py
config.py
types.py
data_io.py
env_adapter.py
feature_engine.py
reward_model.py
src/
core/
scripts/
tests/
```

필요한 helper, dataclass, validation, plotting, CLI는 관련된 기존 파일 내부에 둔다.

---

# 3. 전체 데이터 흐름

```text
1. collect_demo.py
   demo.hdf5 수집
   demo 검사 및 simulator replay
   raw signal 추출
        ↓
   artifacts/raw/raw_demo_XXX.npz

2. feature_select.py
   raw signal에서 feature 계산
   feature normalization
   automatic feature profiling
        ↓
   artifacts/features/features_demo_XXX.npz
   artifacts/features/feature_profiles.json

3. phase_segment.py
   multi-feature boundary evidence
   phase segmentation
   여러 demo의 phase alignment
   change / hold / passive subgoal 추론
        ↓
   artifacts/segmentation/phase_segments.npz
   artifacts/segmentation/canonical_phases.json
   artifacts/segmentation/subgoals.json

4. fcm.py
   baseline / perturbed branch rollout
   future feature residual dataset
   Residual FCM 학습
   candidate ranking 성능 평가
        ↓
   artifacts/fcm/fcm_dataset.npz
   artifacts/fcm/fcm_model.pt 또는 joblib
   artifacts/fcm/fcm_metrics.json

5. degradation.py
   degradation hypothesis 생성
   action candidate 생성
   FCM Top-K screening
   exact simulator validation
   candidate별 lambda calibration
        ↓
   artifacts/degradation/validated_families.npz
   artifacts/degradation/validated_families.json
```

---

# 4. Step 1 — `collect_demo.py` 수정

## 4.1 현재 코드에서 유지하는 부분

현재 collector의 다음 기능은 그대로 유지한다.

- interactive environment 선택
- robot 선택
- OSC_POSE / OSC_POSITION 선택
- keyboard teleoperation
- camera 조작
- `DataCollectionWrapper`
- episode 저장 여부 선택
- accepted demonstration HDF5 저장
- robosuite version 차이를 고려한 controller 및 teleoperation 호출
- environment와 controller metadata 저장

기존 collection 동작은 연구 파이프라인의 시작점이므로 불필요하게 다시 작성하지 않는다.

---

## 4.2 추가할 기능

현재 collector 하단에 다음 실행 mode를 추가한다.

```bash
python collect_demo.py collect
python collect_demo.py inspect --input data/demos
python collect_demo.py extract --input data/demos --output artifacts/raw
```

기존처럼 argument 없이 실행했을 때 interactive collection을 시작하도록 유지해도 된다.

### `collect`

기존 teleoperation loop를 그대로 사용한다.

### `inspect`

HDF5에서 다음을 검사한다.

- states/actions 존재
- trajectory length 일치
- NaN/Inf
- action dimension
- environment name
- robot
- controller
- control frequency
- model XML
- episode 수

### `extract`

각 demo를 simulator에 복원하고 다음 raw signal을 저장한다.

```text
time
states
actions
qpos
qvel
eef_pos
eef_quat
object_pos
object_quat
gripper_aperture
contact
goal_context
```

---

## 4.3 Goal 처리

현재 pipeline에서 goal을 demonstration의 마지막 object position으로 자동 대체하지 않는다.

우선순위:

1. environment의 target site 또는 target region
2. HDF5 metadata에 저장된 goal
3. task-specific resolver
4. 찾을 수 없으면 명확한 오류 또는 해당 goal feature 제외

`goal_context`는 다음처럼 저장할 수 있다.

```json
{
  "type": "target_region",
  "target_pos": [0.1, -0.2, 0.85],
  "position_tolerance": 0.03
}
```

---

## 4.4 출력

```text
artifacts/raw/
├── raw_demo_000.npz
├── raw_demo_000_manifest.json
├── raw_demo_001.npz
└── raw_demo_001_manifest.json
```

---

## 4.5 완료 조건

```text
demo.hdf5 정상 로드
states/actions 길이 일치
actual environment metadata 확인
raw signal의 길이 일치
actual goal context 추출
raw artifact 생성
```

이 조건을 확인하기 전에는 `feature_select.py`를 수정하지 않는다.

---

# 5. Step 2 — `feature_select.py` 수정

## 5.1 현재 코드에서 유지하는 부분

다음 계산 및 helper는 최대한 유지한다.

- quaternion normalization
- relative rotation angle
- angular speed
- position smoothing
- end-effector kinematics
- action perturbation
- clipping fraction
- `compute_trajectory`
- 기존 feature 계산식

현재 구현된 feature들은 검증된 수학식을 다시 작성하지 않고 사용한다.

---

## 5.2 제거하거나 더 이상 사용하지 않을 설계

현재의 `FeatureSpec.kind`와 `FeatureSpec.boundary`를 downstream 알고리즘이 직접 정답처럼 사용하지 않도록 수정한다.

현재와 같은 구조:

```python
FeatureSpec(
    name="eef_object_dist",
    kind="progress",
    boundary=True,
)
```

는 사람이 feature 역할과 phase 경계를 미리 결정한다.

최종 구조에서는 사람은 다음만 정의한다.

```python
FEATURE_FUNCTIONS = {
    "eef_object_dist": compute_eef_object_dist,
    "object_goal_dist": compute_object_goal_dist,
    "contact": compute_contact,
    "eef_jerk": compute_eef_jerk,
}
```

기존 feature list와 계산 함수는 유지하되 `progress`, `event`, `quality`, `boundary`는 automatic profiler의 출력으로 대체한다.

초기 전환 기간에는 기존 `FeatureSpec`을 삭제하지 않고 compatibility 용도로 남길 수 있으나, `phase_segment.py`와 `fcm.py`가 이를 segmentation 정답으로 사용해서는 안 된다.

---

## 5.3 Automatic feature profiler 추가

각 feature 신호에 대해 다음 값을 계산한다.

```text
variation_score
trend_score
transition_score
plateau_score
cross_demo_consistency
structural_score
boundary_score
confidence
```

Profiler는 feature 이름으로 규칙을 만들지 않고 실제 trajectory signal을 분석한다.

### Trend

- 시작과 끝 사이의 net change
- total variation 대비 net change
- smoothing 이후 변화 방향 일관성

### Transition

- 짧은 구간의 급격한 변화
- binary 또는 near-binary mode transition

### Plateau

- 큰 변화 이후 낮은 derivative 상태 유지

### Cross-demo consistency

- 여러 demonstration에서 비슷한 event order와 phase-relative 위치에 변화가 반복되는지 확인

---

## 5.4 Feature 역할에 대한 원칙

Automatic profiler 단계에서는 feature를 영구적인 `progress`, `event`, `quality` label로 고정하지 않는다.

최종 역할은 `phase_segment.py`가 각 phase 내부에서 결정한다.

```text
change
hold
passive
```

같은 feature도 phase마다 다른 역할을 가질 수 있다.

---

## 5.5 실행

```bash
python feature_select.py all \
  --input artifacts/raw \
  --output artifacts/features
```

---

## 5.6 출력

```text
artifacts/features/
├── features_demo_000.npz
├── features_demo_001.npz
├── normalization_stats.json
├── feature_profiles.json
└── feature_manifest.json
```

---

## 5.7 완료 조건

```text
기존 feature 계산 결과 유지
모든 feature shape = (T,)
NaN/Inf 없음
feature order 저장
automatic profile 생성
수동 boundary metadata 없이 다음 단계 실행 가능
```

---

# 6. Step 3 — `phase_segment.py` 수정

## 6.1 현재 코드에서 유지하는 부분

다음 robosuite helper는 유지한다.

- HDF5 demo reading
- environment reconstruction
- model XML restore
- object resolver
- object body resolver
- gripper geometry resolver
- contact signal
- raw frame reading
- replay 및 visualization helper

---

## 6.2 교체할 부분

현재 `segment_features()`는 `feature_select.py`의 `kind`와 `boundary`에 의존한다.

이를 다음 흐름으로 변경한다.

```text
feature_values
+ feature_profiles
→ feature별 boundary evidence
→ weighted multivariate boundary score
→ candidate peaks
→ nearby peak merge
→ minimum phase length
→ latent phases
```

기존 high-water mark와 event transition 함수는 삭제하지 않고 boundary evidence 후보 중 하나로 재사용할 수 있다.

즉:

```text
기존 plateau detector
기존 event transition detector
```

를 버리는 것이 아니라, 수동으로 지정된 특정 feature에만 적용하지 않고 profiler 결과에 따라 적용한다.

---

## 6.3 Multi-demo segmentation

각 demo를 독립적으로 segmentation한 뒤 segment descriptor를 계산한다.

Descriptor 예:

```text
phase duration
feature start value
feature end value
feature net change
feature variance
event transition
```

여러 demo의 segment sequence를 정렬하여 canonical latent phase sequence를 만든다.

초기 구현에서는 복잡한 deep sequence model 대신 dynamic programming 또는 순서 보존 alignment를 사용한다.

---

## 6.4 Subgoal inference

각 canonical phase와 feature마다 다음 역할을 추론한다.

### Change

```text
phase start-end 변화량이 충분함
변화 방향이 phase 내부에서 일관됨
여러 demo에서 direction consistency가 높음
```

저장:

```text
feature
observed_direction
start_distribution
end_distribution
effect_size
confidence
```

### Hold

```text
phase 내부 normalized variance가 낮음
여러 demo에서 reference가 안정적
```

저장:

```text
feature
reference_center
reference_scale
confidence
```

### Passive

Change와 Hold 조건을 안정적으로 만족하지 않는 feature.

Passive feature는 FCM prediction output에는 남지만 degradation target으로 사용하지 않는다.

---

## 6.5 Completion / degradation score

Phase subgoal은 하나의 waypoint가 아니라 Change와 Hold 조건의 집합이다.

단, 초기 degradation family는 해석을 명확하게 하기 위해 **하나의 target feature를 중심으로 생성**한다.

```text
one family
= one phase
+ one change 또는 hold target
+ one action subspace
```

여러 feature를 사람이 정한 가중치로 하나의 objective에 합치지 않는다.

---

## 6.6 Goal fallback 수정

현재의 `infer_goal(obj[-1])` 방식은 제거하거나 명시적인 debug fallback으로만 남긴다.

기본 동작은 `collect_demo.py`가 저장한 `goal_context`를 사용한다.

---

## 6.7 실행

```bash
python phase_segment.py all \
  --features artifacts/features \
  --output artifacts/segmentation
```

---

## 6.8 출력

```text
artifacts/segmentation/
├── phase_segments.npz
├── canonical_phases.json
├── subgoals.json
├── segmentation_metrics.json
└── segmentation_gate.json
```

---

## 6.9 완료 조건

```text
phase 개수와 이름 하드코딩 없음
minimum phase length 만족
여러 demo에 canonical alignment 가능
change/hold/passive 자동 생성
goal final-position fallback 미사용
segmentation output을 FCM이 직접 읽을 수 있음
```

---

# 7. Step 4 — `fcm.py` 수정

## 7.1 현재 코드에서 유지하는 부분

다음 코드는 재사용한다.

- `DemoRollout`의 simulator interaction 구조
- action perturbation 및 clipping helper 호출
- MLP 기반 multi-output FCM
- input/output scaling
- action subspace 분리
- random direction sampling
- candidate scoring helper 일부
- visualization helper

---

## 7.2 FCM 역할 정리

현재 `fcm.py`가 dataset, model training, candidate search, simulator validation, lambda search를 모두 수행하면 `degradation.py`와 책임이 중복된다.

최종 책임은 다음으로 제한한다.

```text
1. counterfactual residual dataset 수집
2. Residual FCM 학습
3. validation prediction metric
4. candidate ranking metric
5. model checkpoint 저장
```

다음은 `degradation.py`로 이동한다.

```text
degradation hypothesis
candidate별 exact simulator validation
lambda_max 결정
degradation ladder 생성
final family 저장
```

기존 함수를 완전히 삭제하기보다 필요한 함수를 `degradation.py`로 옮긴 후 FCM 실행 flow에서 호출하지 않도록 정리한다.

---

## 7.3 Branch rollout 수정

현재 arbitrary MuJoCo state를 set한 뒤 rollout하면 OSC controller의 internal target이 재현되지 않을 수 있다.

최종 방식:

```text
episode initial state
→ anchor까지 demo action replay
→ controller state 재구성
→ baseline branch와 perturbed branch 분기
```

두 branch는 동일한 anchor simulator state와 controller history에서 시작해야 한다.

---

## 7.4 Residual target 수정

현재 perturbed branch의 현재 feature 대비 변화:

```text
feature_perturbed(t+h) - feature_perturbed(t)
```

가 아니라 다음을 사용한다.

\[
Y_{t,h}
=
\phi^{pert}_{t+h}
-
\phi^{base}_{t+h}
\]

여기서 baseline은 동일 anchor에서 demo action을 실행한 actual simulator branch다.

Zero perturbation:

\[
\delta=0
\Rightarrow
Y_{t,h}\approx0
\]

---

## 7.5 FCM 입력 수정

최종 input:

\[
X_{t,h}
=
[
\phi_t,
a_t,
\delta_t,
z_t,
\rho_t,
g,
h
]
\]

추가 항목:

- phase id
- phase progress
- 안정적으로 추출된 경우 goal context
- rollout id
- demo id

---

## 7.6 Dataset split 수정

현재 shuffled row split은 같은 rollout의 인접 sample을 train과 validation에 동시에 넣을 수 있다.

다음 기준을 사용한다.

```text
demo_id split
또는
rollout_id group split
```

동일 rollout의 여러 horizon sample은 같은 split에 들어가야 한다.

---

## 7.7 평가 수정

Prediction metric:

```text
feature-wise MAE
feature-wise R²
phase-wise MAE
horizon-wise MAE
zero-delta residual
```

Screening metric:

```text
Spearman correlation
Top-K recall
best candidate inclusion rate
random screening 대비 향상
```

FCM은 평균 R²만으로 통과시키지 않는다.

---

## 7.8 실행

```bash
python fcm.py collect \
  --raw artifacts/raw \
  --features artifacts/features \
  --phases artifacts/segmentation \
  --output artifacts/fcm

python fcm.py train \
  --dataset artifacts/fcm/fcm_dataset.npz

python fcm.py evaluate \
  --model artifacts/fcm/fcm_model.pt
```

또는:

```bash
python fcm.py all
```

---

## 7.9 출력

```text
artifacts/fcm/
├── fcm_dataset.npz
├── fcm_model.pt
├── fcm_metrics.json
└── fcm_gate.json
```

현재 scikit-learn 모델을 유지한다면 `fcm_model.joblib`을 사용할 수 있다. 중요한 것은 model format보다 residual target과 data split의 일관성이다.

---

## 7.10 완료 조건

```text
baseline-relative residual
zero perturbation check
branch start consistency
group split
phase-conditioned input
Top-K recall 저장
model checkpoint 생성
```

---

# 8. Step 5 — `degradation.py` 수정

## 8.1 현재 코드에서 유지하는 부분

다음 기능은 유지한다.

- phase 구간에 perturbation을 적용하는 action sequence builder
- action clipping
- full trajectory simulator executor
- executed trajectory feature 계산
- lambda별 trajectory 저장
- 3D trajectory 및 feature curve plotting
- lambda=0 reproduction check

---

## 8.2 현재 범위에서 제거할 부분

`make_preference_pairs()`와 preference pair 저장은 현재 확정 범위에서 실행하지 않는다.

이 함수는 후속 reward learning 단계에서 다시 사용할 수 있도록 코드 하단에 보존할 수 있으나, 현재 `main()`의 final output에 포함하지 않는다.

또한 다음 가정을 사용하지 않는다.

```text
lambda가 크면 실제 측정 없이 자동으로 더 나쁨
```

실제 simulator에서 target feature degradation을 측정한 뒤 family를 채택한다.

---

## 8.3 Degradation hypothesis 생성

`subgoals.json`에서 자동 생성한다.

### Reverse change

Demonstration에서 change feature가 움직인 방향의 반대를 목표로 한다.

### Disturb hold

Hold reference distribution에서 벗어나도록 한다.

### Passive

Hypothesis 생성하지 않는다.

한 family는 다음으로 정의한다.

```text
phase id
target feature
hypothesis mode
action subspace
candidate direction
```

---

## 8.4 Candidate generation

Action dimension과 controller layout을 metadata에서 읽는다.

OSC_POSE:

```text
position: 0:3
rotation: 3:6
gripper: 6:7
```

각 subspace에서:

- positive/negative axis
- random unit vector
- 필요하면 기존 geometric seed

를 생성한다.

---

## 8.5 FCM screening

저장된 FCM model을 로드한다.

각 candidate를 여러 anchor와 작은 magnitude에서 평가하고 target feature 기준으로 score를 계산한다.

초기에는 하나의 family가 하나의 target feature를 사용한다.

```text
reverse_change
→ demo direction을 방해하는 predicted residual

disturb_hold
→ reference center에서 멀어지는 predicted residual
```

Top-K candidate만 actual simulator로 넘긴다.

---

## 8.6 Exact simulator validation

각 Top-K candidate를 실제 simulator에서 실행한다.

검사 항목:

```text
lambda=0 original reproduction
task success
target feature actual degradation
clipping fraction
physical feasibility
trajectory completion
```

---

## 8.7 Candidate별 lambda calibration

각 candidate에 대해 별도로 lambda sweep를 수행한다.

```text
candidate A → lambda_max_A
candidate B → lambda_max_B
candidate C → lambda_max_C
```

하나의 candidate에서 얻은 lambda를 다른 candidate에 사용하지 않는다.

Success sequence가:

```text
True, True, False, True
```

이면 처음 failure 전까지의 success prefix만 사용한다.

---

## 8.8 Degradation ladder

\[
\lambda
=
\{
0,
0.33\lambda_{max},
0.66\lambda_{max},
\lambda_{max}
\}
\]

각 level의 actual degradation score를 계산한다.

Accepted family 조건:

```text
task success 유지
actual target degradation 대체로 단조 증가
minimum effect-size 만족
clipping 제한 만족
lambda=0 reproduction 통과
```

---

## 8.9 실행

```bash
python degradation.py hypothesize \
  --subgoals artifacts/segmentation/subgoals.json

python degradation.py screen \
  --model artifacts/fcm/fcm_model.pt

python degradation.py validate \
  --candidates artifacts/degradation/screened_candidates.json

python degradation.py all
```

---

## 8.10 출력

```text
artifacts/degradation/
├── degradation_hypotheses.json
├── screened_candidates.json
├── lambda_search_results.json
├── validated_families.npz
├── validated_families.json
└── degradation_metrics.json
```

---

## 8.11 완료 조건

```text
change/hold hypothesis 자동 생성
passive target 없음
FCM Top-K screening
candidate별 lambda_max
simulator actual degradation 단조성
success 유지
validated family 저장
```

---

# 9. 기존 코드 수정 순서

한 번에 다섯 파일을 동시에 수정하지 않는다.

## 9.1 `collect_demo.py`

기존 collector 실행이 유지되는지 먼저 확인한다.

수정 후:

```text
기존 collect 동작
+
inspect
+
raw extract
```

가 모두 가능해야 한다.

## 9.2 `feature_select.py`

기존 `compute_trajectory()` 결과가 바뀌지 않는 상태에서 profiler를 추가한다.

기존 feature 계산 결과와 수정 후 결과가 동일한지 비교한다.

## 9.3 `phase_segment.py`

기존 replay와 object/contact resolver를 유지하고 segmentation 함수만 단계적으로 교체한다.

초기에는 기존 segmentation 결과와 automatic segmentation 결과를 둘 다 출력하여 비교할 수 있다.

안정화 후 automatic 결과를 default로 전환한다.

## 9.4 `fcm.py`

먼저 target과 branch만 수정한다.

```text
baseline-relative residual
zero-delta consistency
```

가 확인된 뒤 phase input과 group split을 추가한다.

마지막에 Top-K ranking metric을 추가한다.

## 9.5 `degradation.py`

기존 full trajectory executor를 유지한다.

다음 순서로 수정한다.

```text
subgoal hypothesis
→ FCM model load
→ candidate screening
→ exact validation
→ candidate별 lambda
→ family acceptance
```

---

# 10. 기존 코드 보존 규칙

코딩 과정에서 다음 규칙을 적용한다.

1. 기존 파일 이름을 바꾸지 않는다.
2. working function을 이유 없이 삭제하지 않는다.
3. 함수 signature를 바꾸면 호출부를 같은 단계에서 함께 수정한다.
4. 기존 CLI는 가능한 한 유지한다.
5. 새 CLI mode는 기존 `main()`에 subcommand 형태로 추가한다.
6. 기존 output schema를 변경할 때는 새 schema version을 기록한다.
7. 기존 output을 조용히 덮어쓰지 않는다.
8. 단계별로 별도의 output directory를 사용한다.
9. 동작하지 않는 placeholder를 추가하지 않는다.
10. 현재 범위 밖 reward/policy 코드를 추가하지 않는다.

---

# 11. 전체 실행 순서

```bash
# 1. demonstration 수집
python collect_demo.py collect

# 2. demo 검사
python collect_demo.py inspect --input data/demos

# 3. raw signal 추출
python collect_demo.py extract \
  --input data/demos \
  --output artifacts/raw

# 4. feature 계산 및 profiling
python feature_select.py all \
  --input artifacts/raw \
  --output artifacts/features

# 5. phase 및 subgoal
python phase_segment.py all \
  --features artifacts/features \
  --output artifacts/segmentation

# 6. residual FCM
python fcm.py all \
  --raw artifacts/raw \
  --features artifacts/features \
  --phases artifacts/segmentation \
  --output artifacts/fcm

# 7. structured degradation
python degradation.py all \
  --demos data/demos \
  --raw artifacts/raw \
  --features artifacts/features \
  --phases artifacts/segmentation \
  --fcm artifacts/fcm/fcm_model.pt \
  --output artifacts/degradation
```

각 파일이 끝날 때 다음을 출력한다.

```text
입력 경로
처리한 demo 수
핵심 shape
생성한 output
현재 Gate PASS/FAIL
다음 실행 명령
```

---

# 12. 최종 파이프라인

```text
기존 collect_demo.py의 teleoperation 기능을 유지한다.
→ 같은 파일에 demo 검사와 raw replay 기능을 추가한다.
→ 기존 feature_select.py의 계산식을 유지하고 automatic profiler를 추가한다.
→ 기존 phase_segment.py의 replay helper를 유지하고 segmentation/subgoal 알고리즘을 교체한다.
→ 기존 fcm.py의 rollout과 MLP를 유지하고 baseline-relative residual FCM으로 수정한다.
→ 기존 degradation.py의 executor를 유지하고 hypothesis, screening, lambda calibration을 통합한다.
→ simulator에서 검증된 degradation family만 저장한다.
```

최종 산출물:

```text
ValidatedDegradationFamilySet
```

현재 단계에서 주장하는 연구 결과:

> 성공하지만 비효율적인 demonstration의 feature trajectory에서 안정적인 latent phase와 phase-local subgoal을 추론하고, residual FCM으로 action perturbation 후보를 선별한 뒤 actual simulator validation을 통해 성공을 유지하면서 점진적으로 열화되는 structured degradation family를 생성한다.

현재 단계에서 주장하지 않는 결과:

> 학습된 policy가 demonstrator보다 우수하다.