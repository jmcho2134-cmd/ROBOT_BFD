# 성공하지만 비효율적인 시연으로부터의 자동 구조화 열화 생성  
## Automatic Phase-Conditioned Structured Degradation from Suboptimal Demonstrations via Residual Forward Consequence Modeling

---

## 초록

로봇 모방학습은 전문가 시연의 행동을 재현하는 데 효과적이지만, 시연이 성공적이면서도 우회 동작, 불필요한 정렬, 정지, 과도한 회전과 같은 비효율성을 포함하는 경우 정책 역시 그 한계를 그대로 모방할 수 있다. 기존의 D-REX 및 SSRR 계열 방법은 시연 또는 정책에 무작위 노이즈를 주입하여 다양한 품질의 궤적을 생성하고, 이들 사이의 상대적 순위를 통해 보상함수를 학습한다. 그러나 단순한 Gaussian noise는 로봇 조작 과제의 의미적 구조를 반영하지 못하며, 동일한 크기의 노이즈도 현재 수행 중인 부분동작에 따라 전혀 다른 결과를 초래할 수 있다.

본 연구는 성공하지만 비효율적인 suboptimal demonstration으로부터 **부분동작의 시간적 구조를 자동으로 추론하고**, 각 부분동작의 subgoal을 방해하는 **phase-conditioned structured degradation**을 생성하는 방법을 제안한다. 연구자는 feature의 의미적 역할이나 선호 방향을 직접 지정하지 않고, simulator state로부터 계산 가능한 feature 함수만 정의한다. 제안 방법은 feature 신호의 변화성, 전이성, 정체성 및 일관성을 분석하여 latent phase를 분할하고, 각 phase에서 feature를 change, hold, passive 역할로 분류한다. 이후 Residual Forward Consequence Model(FCM)은 현재 feature, demonstration action, action perturbation, phase 및 phase progress를 입력받아 perturbation에 의해 발생할 미래 feature residual을 예측한다. FCM은 최종 품질 판정기가 아니라 다수의 action perturbation 후보를 선별하는 screening model로 사용되며, 최종 degradation 방향과 크기는 반드시 실제 simulator rollout을 통해 검증된다.

현재 프로젝트의 직접적인 구현 목표는 reward learning 이전 단계인 **Validated Degradation Family의 생성**이다. 각 family는 동일한 initial state, 동일한 phase, 동일한 target subgoal을 공유하면서 original–mild–medium–severe 수준으로 열화가 증가하고, task success와 물리적 실행 가능성이 유지되는 궤적 집합이다. 장기적으로는 이 validated family를 이용해 preference reward를 학습하고, demonstration 주변에서 보수적으로 정책을 개선함으로써 Better-than-Demonstrator 가능성을 실험적으로 검증한다.

---

# 1. 연구 배경

## 1.1 성공적인 시연이 항상 좋은 시연은 아니다

사람이 teleoperation으로 수집한 로봇 조작 시연은 task success를 달성하더라도 다음과 같은 비효율성을 포함할 수 있다.

- 목표물로 곧바로 접근하지 않고 우회하는 sidetrack
- 물체를 잡기 전 불필요하게 반복되는 자세 정렬
- 이동 중의 정지 또는 되돌아가기
- 과도한 end-effector 회전
- 물체를 목표 지점 주변에서 반복적으로 재배치
- 불필요하게 긴 경로와 큰 action variation

Behavioral Cloning은 이러한 행동까지 데이터 분포의 일부로 학습한다. 따라서 시연자가 성공했다는 사실만으로 학습된 정책이 효율적이라고 보장할 수 없다.

본 프로젝트의 출발점은 다음과 같다.

> 성공한 시연을 그대로 정답으로 간주하지 않고, 시연을 구성하는 부분동작의 의도와 subgoal을 추론한 뒤, 그 subgoal을 체계적으로 방해하는 궤적을 생성하면 시연의 “무엇이 중요한가”를 더 명확하게 학습할 수 있다.

## 1.2 무작위 열화의 한계

D-REX 및 SSRR 계열 접근은 무작위 노이즈의 크기를 증가시켜 원본보다 점차 나쁜 궤적을 생성하고, 궤적 순위를 이용해 보상을 학습한다. 이 접근은 별도의 수작업 reward 없이 상대적 품질 정보를 생성할 수 있다는 장점이 있다.

그러나 로봇 조작에서 noise의 결과는 현재 phase에 강하게 의존한다.

- 물체에 접근하는 중의 위치 오차는 sidetrack이 될 수 있다.
- grasp 시점의 작은 회전 오차는 grasp failure 또는 slip을 만들 수 있다.
- transport 중의 gripper 변화는 물체 낙하를 유발할 수 있다.
- place 시점의 위치 오차는 final goal error를 직접 증가시킨다.

따라서 동일한 Gaussian noise라도 어느 시점에 어떤 action dimension에 적용되는지에 따라 의미가 달라진다. 본 연구는 random degradation을 **phase-conditioned structured degradation**으로 대체한다.

---

# 2. 연구 문제 정의

## 2.1 입력

본 연구의 입력은 simulator에서 수집한 successful-but-inefficient demonstration 집합이다.

\[
\tau^{(n)}
=
\left\{
s_t^{(n)}, a_t^{(n)}
\right\}_{t=0}^{T_n-1}
\]

여기서 \(s_t\)는 MuJoCo simulator state, \(a_t\)는 robot controller action, \(T_n\)은 trajectory length다.

초기 구현 환경은 다음과 같다.

- Simulator: robosuite / MuJoCo
- Task: PickPlaceBread
- Robot: UR5e
- Controller: OSC_POSE

\[
a_t =
[
\Delta x,\Delta y,\Delta z,
\Delta r_x,\Delta r_y,\Delta r_z,
g
]
\in \mathbb{R}^{7}
\]

## 2.2 현재 구현 목표

현재 프로젝트는 다음 결과를 생성하는 것을 직접적인 목표로 한다.

\[
\mathcal{F}_{deg}
=
\left\{
\tau^{orig},
\tau^{mild},
\tau^{medium},
\tau^{severe}
\right\}
\]

각 degradation family는 다음 조건을 만족해야 한다.

1. 동일한 demonstration 및 initial condition에서 생성된다.
2. 동일한 latent phase와 동일한 subgoal feature를 대상으로 한다.
3. 동일한 degradation direction을 공유하고 magnitude만 달라진다.
4. task success가 가능한 범위에서 유지된다.
5. target subgoal degradation이 magnitude에 따라 대체로 단조 증가한다.
6. action clipping과 물리적 비현실성이 제한된다.

현재 범위는 degradation family 생성까지이며, reward learning과 policy improvement는 후속 연구 단계로 둔다.

---

# 3. 핵심 연구 질문

### RQ1. Feature 계산 함수만으로 비효율적인 시연의 시간적 구조를 자동으로 분할할 수 있는가?

연구자가 Approach, Grasp, Transport와 같은 phase label을 직접 입력하지 않고도, feature 신호의 변화와 전이를 이용해 의미 있는 latent segment를 얻을 수 있는지를 평가한다.

### RQ2. 각 phase의 subgoal을 feature의 변화 및 유지 조건으로 자동 표현할 수 있는가?

각 phase에서 어떤 feature가 목표 방향으로 변화해야 하는지(change), 어떤 feature가 안정적으로 유지되어야 하는지(hold), 어떤 feature가 직접적인 구조적 역할을 갖지 않는지(passive)를 자동 추론한다.

### RQ3. Residual FCM이 subgoal을 가장 크게 방해할 action perturbation 후보를 효율적으로 선별할 수 있는가?

FCM이 전체 next state를 예측하는 대신 perturbation으로 인한 future feature residual을 예측할 때, 실제 simulator에서 강한 degradation을 만드는 후보를 Top-K 안에 포함할 수 있는지를 평가한다.

### RQ4. FCM과 exact simulator validation을 결합하면 성공을 유지하는 단계적 degradation family를 생성할 수 있는가?

FCM prediction을 최종 label로 사용하지 않고 simulator rollout으로 검증함으로써 model error에 의한 잘못된 degradation을 줄일 수 있는지를 확인한다.

### RQ5. 장기적으로 structured degradation이 random degradation보다 더 안정적인 preference signal을 제공하는가?

이는 reward learning 확장 단계에서 검증할 질문이다.

---

# 4. 연구 가설

### H1. 자동 phase segmentation
여러 feature의 boundary evidence를 결합하면 단일 feature 또는 고정 timestep 기반 segmentation보다 다양한 시연에서 안정적인 latent phase를 얻을 수 있다.

### H2. Phase-local subgoal
각 phase에서 추론된 change 및 hold feature는 해당 phase의 진행과 안정성을 표현하며, passive feature보다 degradation target으로서 높은 일관성을 보인다.

### H3. FCM screening
Residual FCM을 사용한 Top-K candidate screening은 random candidate selection보다 실제 simulator degradation score가 높은 후보를 더 자주 포함한다.

### H4. Simulator-validated monotonicity
Candidate별로 독립적인 magnitude calibration을 수행하면 임의의 고정 magnitude를 사용하는 것보다 successful and monotonic degradation family의 생성 비율이 증가한다.

### H5. 장기적 Better-than-Demonstrator 가능성
Validated structured degradation에서 학습한 preference reward는 suboptimal demonstration 주변에서 local policy improvement를 유도할 가능성이 있다. 단, 이는 이론적 보장이 아니라 실험적으로 검증할 가설이다.

---

# 5. 제안 방법 개요

```text
Phase 0: Demonstration Collection
    ↓
Phase 1: Demo Understanding
    Feature Computation
    Automatic Feature Profiling
    Phase Segmentation
    Subgoal Inference
    ↓
Phase 2: Residual Forward Consequence Model
    Perturbation Dataset
    Residual Prediction
    Candidate Ranking
    ↓
Phase 3: Structured Degradation
    Hypothesis Generation
    Candidate Search
    Magnitude Calibration
    Exact Simulator Validation
```

# 6. Phase 0 — Successful-but-Inefficient Demonstration Collection

## 6.1 목적

Phase 0의 목적은 단순히 성공 궤적을 많이 수집하는 것이 아니라, 성공 조건을 만족하면서도 다양한 형태의 비효율성을 포함하는 demonstration을 수집하는 것이다.

## 6.2 수집 방법

사용자는 keyboard teleoperation을 이용하여 robosuite의 UR5e를 OSC_POSE controller로 조작한다. 각 trajectory는 simulator state와 controller action의 sequence로 저장한다.

저장 항목:

- simulator states
- actions
- environment configuration
- controller configuration
- MuJoCo model XML
- robot 및 task metadata
- control frequency

## 6.3 Demonstration 다양성

정식 실험에서는 다음 조건을 변화시킨다.

- initial object position
- target position
- random seed
- trajectory length
- inefficiency type
- inefficiency severity
- 사용자 수행 방식

초기 pilot에서는 5–10개의 성공 시연으로 코드와 지표를 검증하고, 본 실험에서는 조건별 반복을 포함해 20개 이상의 demonstration을 목표로 한다.

## 6.4 데이터 품질 기준

- task success
- state와 action의 길이 일치
- NaN/Inf 없음
- action dimension과 controller configuration 일치
- 지나치게 짧거나 불완전한 episode 제외
- 수집자가 의도적으로 포함한 inefficiency 메모 기록

Phase 0의 출력은 replay 가능한 `demo.hdf5`다.

---

# 7. Phase 1 — Automatic Demo Understanding

## 7.1 Raw signal extraction

Simulator replay를 통해 각 timestep에서 다음 raw signal을 추출한다.

- end-effector position and orientation
- object position and orientation
- goal position
- gripper aperture
- gripper–object contact
- joint position and velocity
- action
- control time

\[
r_t
=
[
p^{ee}_t,
R^{ee}_t,
p^{obj}_t,
R^{obj}_t,
p^{goal},
g_t,
contact_t,
q_t,
\dot q_t,
a_t
]
\]

## 7.2 Feature bank

연구자가 입력하는 정보는 feature 이름과 계산 함수뿐이다.

\[
\phi_t=f(r_{0:t})\in\mathbb{R}^{F}
\]

초기 feature bank:

### 관계 및 task-progress signal

\[
d_{ee,obj}(t)=\|p^{ee}_t-p^{obj}_t\|_2
\]

\[
d_{obj,goal}(t)=\|p^{obj}_t-p^{goal}\|_2
\]

\[
h_{obj}(t)=p^{obj}_{t,z}
\]

- eef–object distance
- object–goal distance
- object height
- grasp alignment

### Event 및 interaction signal

- gripper aperture
- contact
- grasp proxy
- release transition

### Motion 및 execution signal

- end-effector speed
- end-effector acceleration
- end-effector jerk
- object speed
- action magnitude
- path increment
- object slip

Registry에는 progress/event/quality label, boundary 여부, higher-is-better 또는 higher-is-worse, phase 이름, degradation 방향을 수동 입력하지 않는다.

## 7.3 Feature normalization

\[
\tilde{\phi}_{t,j}
=
\frac{
\phi_{t,j}-\operatorname{median}(\phi_{\cdot,j})
}{
\operatorname{IQR}(\phi_{\cdot,j})+\epsilon
}
\]

여러 demonstration을 사용할 경우 normalization 통계는 training demonstration 집합에서 계산하고 고정한다.

## 7.4 Automatic feature profiling

각 feature 신호에 대해 다음 soft score를 계산한다.

- variation score
- trend consistency
- transition strength
- plateau strength
- cross-demonstration consistency
- structural confidence
- boundary confidence

Profiler는 feature를 단일 hard category로 고정하지 않는다. 동일한 feature가 어떤 phase에서는 change 역할을 하고 다른 phase에서는 hold 또는 passive가 될 수 있기 때문이다.

## 7.5 Multivariate phase segmentation

Feature별 boundary evidence를 결합한다.

\[
B_t=\sum_{j=1}^{F}w_jb_{t,j}
\]

- \(b_{t,j}\): feature \(j\)의 timestep \(t\) boundary evidence
- \(w_j\): structural confidence 기반 가중치

Boundary evidence에는 smoothed temporal derivative, event transition, local distribution change, plateau onset, multi-feature agreement가 포함된다.

최종 phase:

\[
z_t\in\{z_0,z_1,\dots,z_{K-1}\}
\]

Approach, Grasp, Transport와 같은 의미 label은 사전에 강제하지 않는다.

## 7.6 Phase-local subgoal inference

### Change feature

\[
\Delta\phi_{z,j}
=
\phi_{t^{end}_z,j}
-
\phi_{t^{start}_z,j}
\]

충분한 start–end 차이, 높은 trend consistency, 여러 demo에서의 방향 일관성을 가지면 change로 분류한다.

\[
G^{change}_{z,j}
=
(
\operatorname{sign}(\Delta\phi_{z,j}),
\mu^{start}_{z,j},
\mu^{end}_{z,j},
\sigma_{z,j}
)
\]

### Hold feature

Phase 내부 normalized variance가 낮고 일정 reference를 유지하면 hold로 분류한다.

\[
G^{hold}_{z,j}
=
(
\mu_{z,j},
\sigma_{z,j}
)
\]

### Passive feature

Change 또는 hold 조건을 일관되게 만족하지 못하면 passive로 둔다.

Passive feature는 제거하지 않는다. FCM은 모든 feature residual을 예측하지만 Phase 3에서 degradation target으로 직접 사용하지 않는다.

## 7.7 Phase 1 출력

- feature matrix \(\Phi\in\mathbb{R}^{T\times F}\)
- feature names
- boundary scores
- boundary indices
- phase id \(z_t\)
- phase별 change feature
- phase별 hold feature
- phase별 passive feature

---

# 8. Phase 2 — Residual Forward Consequence Model

## 8.1 목적

FCM의 목적은 다음 simulator state 전체를 정확히 예측하는 것이 아니라, 특정 action perturbation이 미래 feature trajectory를 demonstration baseline에 비해 어떻게 변화시키는지를 예측하는 것이다.

## 8.2 Branch rollout 생성

Anchor timestep \(t\)를 선택하고 episode 시작부터 \(t\)까지 demonstration action을 replay한다. 이는 MuJoCo state뿐 아니라 OSC controller 내부 target과 동적 상태를 일관되게 재구성하기 위해 필요하다.

Baseline branch:

\[
a^{base}_{t:t+h-1}=a^{demo}_{t:t+h-1}
\]

Perturbed branch:

\[
a^{pert}_t=
\operatorname{clip}(a^{demo}_t+\delta_t)
\]

## 8.3 Residual target

\[
y_{t,h}
=
\phi^{pert}_{t+h}
-
\phi^{base}_{t+h}
\]

Zero perturbation 조건:

\[
\delta_t=0
\Rightarrow
y_{t,h}\approx0
\]

## 8.4 FCM 입력

\[
x_{t,h}
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

Phase progress:

\[
\rho_t
=
\frac{
t-t^{start}_{z_t}
}{
t^{end}_{z_t}-t^{start}_{z_t}
}
\]

## 8.5 모델

\[
\hat y_{t,h}=F_{\eta}(x_{t,h})
\]

\[
\mathcal{L}_{FCM}
=
\frac{1}{N}
\sum_{i=1}^{N}
\|\hat y_i-y_i\|_2^2
\]

초기 모델은 multi-output MLP를 사용한다. Input과 target normalization 통계를 checkpoint에 저장한다.

## 8.6 Data split

동일 rollout의 여러 horizon sample이 train과 validation에 동시에 들어가지 않도록 rollout 또는 demonstration 단위 group split을 사용한다.

## 8.7 평가

Prediction metric:

- feature별 MAE
- feature별 \(R^2\)
- phase별 error
- horizon별 error

Screening metric:

- Spearman rank correlation
- Top-K recall
- best candidate inclusion rate

FCM의 유용성은 모든 feature를 완벽히 예측하는지보다 실제로 강한 degradation 후보를 Top-K에 포함하는지로 판단한다.

---

# 9. Phase 3 — Structured Degradation

## 9.1 Degradation hypothesis generation

### Reverse-change hypothesis

\[
H^{reverse}_{z,j}:
\quad
\operatorname{sign}
(
\Delta\phi^{pert}_{z,j}
)
=
-
\operatorname{sign}
(
\Delta\phi^{demo}_{z,j}
)
\]

예:

- eef–object distance 감소 phase → 거리 감소를 방해
- object–goal distance 감소 phase → goal progress 감소
- object height 증가 phase → vertical progress 감소

### Disturb-hold hypothesis

\[
H^{hold}_{z,j}
:
\quad
\frac{
|\phi^{pert}_{z,j}-\mu_{z,j}|
}{
\sigma_{z,j}+\epsilon
}
\text{ 증가}
\]

Passive feature에서는 hypothesis를 만들지 않는다.

## 9.2 Action candidate generation

\[
\mathcal{A}
=
\mathcal{A}_{pos}
\oplus
\mathcal{A}_{rot}
\oplus
\mathcal{A}_{grip}
\]

OSC_POSE:

- position: indices \(0:3\)
- rotation: indices \(3:6\)
- gripper: index \(6\)

각 subspace에서 positive/negative axis direction, random unit direction, zero baseline을 생성한다.

## 9.3 FCM-based candidate screening

Reverse-change score:

\[
S^{reverse}_{z,j}(\delta^k)
=
-
d_{z,j}
\hat y_{z,j}(\delta^k)
\]

Disturb-hold score:

\[
S^{hold}_{z,j}(\delta^k)
=
\frac{
|
\phi_{z,j}
+
\hat y_{z,j}(\delta^k)
-
\mu_{z,j}
|
}{
\sigma_{z,j}+\epsilon
}
\]

여러 anchor와 작은 magnitude에서 score를 평균하여 Top-K를 선택한다. 하나의 degradation family는 하나의 structural target feature를 중심으로 정의한다.

## 9.4 Exact simulator validation

FCM Top-K candidate를 실제 simulator에서 실행하고 다음을 확인한다.

- task success
- target subgoal degradation
- action clipping
- physical feasibility
- trajectory stability

FCM prediction만으로 family를 채택하지 않는다.

## 9.5 Candidate-specific magnitude calibration

\[
a'_t=\operatorname{clip}(a_t+\lambda d^k)
\]

각 candidate에 대해 magnitude를 독립적으로 증가시킨다. Success pattern에서 처음 failure 이전까지의 longest success prefix를 사용한다.

예:

```text
[success, success, failure, success]
```

이면 세 번째 이후는 calibration에서 제외한다.

## 9.6 Degradation ladder

\[
\lambda
\in
\{
0,\,
0.33\lambda_{max},\,
0.66\lambda_{max},\,
\lambda_{max}
\}
\]

- original
- mild
- medium
- severe

## 9.7 Family acceptance criterion

Success constraint:

\[
Success(\tau_{\lambda_i})=1
\]

Monotonic degradation:

\[
D_f(\tau_{\lambda_0})
\le
D_f(\tau_{\lambda_1})
\le
D_f(\tau_{\lambda_2})
\le
D_f(\tau_{\lambda_3})
\]

추가 조건:

- 최소 effect-size
- clipping fraction 제한
- physical feasibility
- trajectory reproducibility

## 9.8 최종 출력 예시

```json
{
  "phase_id": 2,
  "target_feature": "object_goal_dist",
  "mode": "reverse_change",
  "action_subspace": "position",
  "direction": [0.7, -0.2, 0.68, 0, 0, 0, 0],
  "lambda_values": [0.0, 0.06, 0.12, 0.18],
  "actual_degradation_scores": [0.0, 0.14, 0.31, 0.52],
  "success_flags": [true, true, true, true],
  "clip_fractions": [0.0, 0.0, 0.01, 0.03]
}
```

# 10. 장기 확장 — Preference Reward Learning

Validated family 내부에서만 다음 순위를 만든다.

\[
\tau^{orig}
\succ
\tau^{mild}
\succ
\tau^{medium}
\succ
\tau^{severe}
\]

서로 다른 phase나 target feature를 가진 family 사이에는 직접적인 순위를 강제하지 않는다.

Reward network 입력:

\[
\psi_t
=
[
\phi_t,
a_t,
z_t,
\rho_t,
g
]
\]

Phase-normalized return:

\[
J_{\theta}(\tau)
=
\sum_{z}
\frac{1}{T_z}
\sum_{t:z_t=z}
R_{\theta}(\psi_t)
\]

Pairwise ranking loss:

\[
\mathcal{L}_{rank}
=
-
\sum_{(i,j)}
w_{ij}
\log
\sigma
(
J_{\theta}(\tau_i)
-
J_{\theta}(\tau_j)
)
\]

\(w_{ij}\)는 simulator에서 확인한 monotonicity와 effect size를 반영한다.

실패 궤적과 성공 궤적 사이의 ranking은 성공 궤적 내부의 efficiency ranking과 분리한다.

---

# 11. 장기 확장 — Conservative Policy Improvement

Original보다 나쁜 데이터만으로 original보다 좋은 영역의 reward shape는 식별되지 않는다. 따라서 Better-than-Demonstrator는 보장이 아니라 empirical hypothesis로 다룬다.

초기 정책:

\[
\pi_{BC}
=
\arg\min_{\pi}
\mathbb{E}_{(s,a)\sim\mathcal{D}}
[
-\log\pi(a|s)
]
\]

Conservative improvement:

\[
\max_{\pi}
\;
\mathbb{E}_{\tau\sim\pi}
[
J_{\theta}(\tau)
]
-
\alpha
D_{KL}
(
\pi(\cdot|s)
\|
\pi_{BC}(\cdot|s)
)
\]

후보 알고리즘:

- KL-regularized SAC
- AWAC
- behavior-regularized offline-to-online RL

목표는 global optimum이 아니라 demonstration 주변의 local improvement다.

---

# 12. 실험 설계

## 12.1 환경

Primary environment:

- robosuite PickPlaceBread
- UR5e
- OSC_POSE
- MuJoCo physics

추가 task는 초기 결과가 안정된 이후 Lift, PickPlaceCan, Stack 중 하나를 고려한다.

## 12.2 데이터 분할

- Training demonstrations
- Validation demonstrations
- Held-out initial positions
- Held-out goal positions
- Held-out random seeds

동일 demonstration에서 생성된 rollout이 train과 test에 동시에 들어가지 않도록 한다.

## 12.3 Phase 1 평가

- demo 간 normalized boundary location variance
- feature 제거에 대한 boundary 변화
- smoothing parameter 변화에 대한 phase 수 안정성
- 너무 짧은 phase 비율
- change direction consistency
- hold variance
- cross-demo role agreement
- passive feature 비율

소수의 사람이 표시한 coarse boundary는 evaluation reference로만 사용할 수 있다.

## 12.4 Phase 2 평가

- per-feature MAE
- per-feature \(R^2\)
- horizon별 error
- phase별 error
- Spearman correlation
- Top-1/Top-3/Top-5 recall
- 실제 best candidate inclusion rate
- random screening 대비 개선

## 12.5 Phase 3 평가

Family yield:

\[
Yield
=
\frac{
\text{accepted validated families}
}{
\text{attempted hypotheses}
}
\]

Monotonicity rate:

\[
M
=
\frac{
\#\text{monotonic families}
}{
\#\text{validated candidate families}
}
\]

추가 평가:

- 각 degradation level의 task success
- target degradation effect size
- clipping fraction
- exhaustive simulator search 대비 rollout 절감률

---

# 13. Baseline 및 Ablation

## 13.1 Random Gaussian degradation

Phase와 subgoal을 고려하지 않고 action에 Gaussian noise를 적용한다.

## 13.2 Random structured direction

Phase와 action subspace는 사용하지만 FCM 대신 random direction을 선택한다.

## 13.3 No-phase FCM

Phase id와 phase progress를 FCM 입력에서 제거한다.

## 13.4 No-subgoal-role

Change/hold/passive 분류 없이 모든 feature를 target 후보로 사용한다.

## 13.5 No-simulator-validation

FCM predicted score만으로 family를 채택한다.

## 13.6 Fixed shared magnitude

모든 candidate에 동일한 \(\lambda\) ladder를 사용한다.

---

# 14. 예상되는 핵심 기여

## Contribution 1. Feature-function-only demo understanding

연구자가 phase label, feature role, good/bad direction을 직접 지정하지 않고 feature 계산 함수만 제공하는 구조.

## Contribution 2. Phase-local subgoal representation

Subgoal을 fixed waypoint가 아니라 change direction과 hold distribution의 조합으로 표현.

## Contribution 3. Residual consequence screening

전체 next state가 아니라 baseline-relative future feature residual을 예측.

## Contribution 4. Model-screened, simulator-validated degradation

FCM은 후보 수를 줄이는 데만 사용하고 최종 품질 판단은 simulator rollout으로 수행.

## Contribution 5. Nested successful degradation families

동일 initial state, phase, target feature 및 direction을 유지하면서 성공 가능한 범위 안에서 단계적으로 열화된 family 생성.

---

# 15. 주요 위험과 대응 전략

## 15.1 Segmentation이 noise에 과민

대응:

- robust smoothing
- minimum phase duration
- nearby peak merge
- multi-feature agreement
- cross-demo consistency
- boundary confidence threshold

## 15.2 Goal 정보 부재

Final object pose를 goal로 대체하지 않는다. Environment replay에서 actual target site 또는 bin을 읽고, 실패 시 해당 feature를 제외하거나 명확히 오류를 보고한다.

## 15.3 Controller internal state 불일치

Episode 시작부터 anchor까지 action을 replay하고 zero perturbation residual 및 branch start-state consistency를 검사한다.

## 15.4 FCM prediction 부정확

- short horizon
- phase-conditioned input
- residual target
- Top-K screening으로 역할 제한
- exact simulator validation

## 15.5 Non-monotonic degradation

- candidate-specific sweep
- longest success prefix
- actual target monotonicity 검사
- non-monotonic family rejection
- clipping guard

## 15.6 Quality feature 방향 불명확

현재 degradation target으로 사용하지 않고 모든 rollout에 기록한다. 장기 preference reward 학습에서 중요도와 방향을 학습한다.

## 15.7 Better-than-Demonstrator extrapolation

- BfD를 empirical hypothesis로 주장
- BC 주변의 conservative policy improvement
- held-out simulator validation
- success 및 goal-error non-inferiority constraint

---

# 16. 성공 기준

## 현재 degradation-only 단계

1. 여러 successful-but-inefficient demo에서 latent phase가 안정적으로 생성된다.
2. Phase별 change/hold/passive subgoal이 자동 추론된다.
3. FCM Top-K recall이 random screening보다 높다.
4. 실제 simulator에서 successful degradation family가 생성된다.
5. Accepted family의 target degradation이 mild–medium–severe에 따라 단조 증가한다.
6. Exhaustive search보다 적은 simulator rollout으로 유효 후보를 찾는다.

## 장기 BfD 단계

- success-rate 감소가 허용 범위 이내
- final goal error non-inferiority
- path length, time, jerk/action cost 중 최소 두 지표 개선
- contact stability와 object drift의 악화 제한
- 여러 random seed에서 평균과 신뢰구간 보고

---

# 17. 연구 일정 예시

## 1단계 — Demo 및 Phase 1 안정화

- multiple demonstration 수집
- raw signal extraction
- feature bank 검증
- segmentation 및 subgoal 분석

## 2단계 — Residual FCM

- branch rollout
- zero perturbation consistency
- short-horizon dataset
- MLP training 및 screening metric

## 3단계 — Structured Degradation

- hypothesis 생성
- action candidate search
- candidate별 magnitude calibration
- validated family 생성

## 4단계 — 실험 및 Ablation

- Gaussian degradation 비교
- random structured direction 비교
- no-phase/no-FCM ablation
- search efficiency 및 family yield 분석

## 5단계 — 후속 연구

- preference reward
- conservative policy improvement
- Better-than-Demonstrator evaluation

---

# 18. 논문 포지셔닝

가장 강하고 안전한 1차 주장은 다음과 같다.

> Phase-conditioned structured degradation은 무작위 action noise보다 로봇 조작의 부분동작과 subgoal을 더 잘 반영하며, 성공을 유지하면서 단계적으로 열화된 궤적 family를 더 안정적으로 생성한다.

2차 주장:

> Residual FCM을 candidate screening에 사용하고 simulator validation을 결합하면, exhaustive simulator search보다 효율적으로 meaningful degradation direction을 발견할 수 있다.

Better-than-Demonstrator는 제한적으로 주장한다.

> Structured degradation에서 학습한 preference signal이 특정 조작 task와 demonstration 분포 주변에서 local policy improvement로 일반화되는지를 실험적으로 검증한다.

---

# 19. 결론

본 프로젝트는 성공하지만 비효율적인 demonstration을 그대로 모방하는 대신, demonstration 내부의 시간적 구조와 phase-local subgoal을 자동 추론하고 이를 체계적으로 방해하는 structured degradation을 생성한다. 연구자가 feature 역할이나 선호 방향을 직접 설계하지 않고 feature 계산 함수만 정의한다는 점에서 기존의 수작업 phase 및 degradation 설계보다 자동화된 구조를 지향한다.

Residual FCM은 action perturbation이 future feature에 미칠 영향을 예측하여 candidate search를 줄이지만, 최종 품질 판단은 actual simulator rollout에 맡긴다. 이를 통해 model error가 직접 preference label로 전파되는 위험을 줄인다. 현재 연구의 직접적인 결과물은 validated degradation family이며, 이 family는 후속 preference reward learning과 conservative policy improvement의 신뢰 가능한 데이터 기반이 된다.

궁극적으로 본 연구는 “나쁜 행동을 무작위로 생성하는 것”이 아니라, “현재 부분동작의 목적을 이해하고 그 목적을 구조적으로 방해하는 것”을 통해 suboptimal demonstration의 한계를 학습하는 방향을 제시한다.
