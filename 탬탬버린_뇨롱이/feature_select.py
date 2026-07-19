#!/usr/bin/env python
"""
feature_select.py — feature 스키마. 이 프로젝트에서 사람이 정하는 유일한 부분.
================================================================================

다운스트림(phase_segment.py, fcm.py)은 전부 이 파일에서 feature 목록/종류/계산을
import 한다. feature 를 정의하는 곳은 여기 하나뿐이다.

사람이 정하는 것 (그리고 이것만)
--------------------------------
feature 하나당 세 가지: NAME, KIND(progress/event/quality), BOUNDARY(phase 경계
자격). 임계값·subgoal 창·degradation 방향/크기는 전부 다운스트림에서 데이터로 나온다.

경계(BOUNDARY) 설계 — image 2 의 5단계를 만드는 조합
---------------------------------------------------
    eef_object_dist  (progress) boundary=True   -> approach 끝(도착)
    object_goal_dist (progress) boundary=True   -> transport 끝(도착)
    gripper_open     (event)    boundary=True   -> grasp(닫힘) / place(열림)
그래서 경계는 { eef 도착, 그리퍼 닫힘, goal 도착, 그리퍼 열림 } 4개 -> 5 phase
(approach / grasp / transport / place / retreat) 가 저절로 나온다.

왜 그리퍼를 경계로 되돌렸나 (near-binary 인데도)
-----------------------------------------------
"그리퍼로 phase 를 자르면 그 phase 를 degrade 하는 길이 '그리퍼 안 닫음 = task 실패'
뿐이라 사다리가 안 생긴다"는 우려가 있었다. 그러나 이 우려는 fcm.py 의 채점이 이미
해결한다: fcm 은 phase 를 그 phase 의 main 이 아니라 subgoal SET 전체로 채점한다.

    D_z = Σ_{f in change} sign·ΔΦ_f/σ_f   (몰아간 것 반대로)
        + Σ_{f in hold}      |ΔΦ_f|/σ_f    (유지한 것 흔들기)

grasp 의 main 이 near-binary 그리퍼여도, 같은 phase 가 contact 를 change,
grasp_align / eef_object_dist 를 hold 하므로 그 연속 feature 로 degrade = 미정렬/
접촉 상실(Sec 8.1) 이 되어 사다리가 정상적으로 선다. 즉 사다리 문제는 세그먼트가
아니라 채점에서 해결되므로, 그리퍼를 경계로 써도 무방하고 오히려 grasp/place 를
깔끔히 분리해 준다.

기타 결정
---------
* contact: event, boundary=False. subgoal 멤버(파지 종료 조건)일 뿐 경계 아님.
* grasp_align: progress, boundary=False. 빵이 비축대칭이라 approach 동안 각이 떨어짐.
  orientation 을 subgoal 로 실어 grasp 를 미정렬로 degrade 하게 함.
* object_height: 기본 quality. 아래 OBJECT_HEIGHT_AS_SUBGOAL 로 progress(boundary=
  False) 전환 가능(그래도 lift phase 는 안 생김).
* waypoint feature 없음: 데모 자기 waypoint 와의 거리는 Sec 9.3 similarity trap.
* perturb_action / clip_fraction / compose_axis_angle 은 M3(fcm.py)가 쓰는
  공용 유틸이라 여기 둔다.

    python feature_select.py --selftest
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass

import numpy as np

try:
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation as _R
    _HAS_SCIPY = True
except Exception:                                             # pragma: no cover
    _HAS_SCIPY = False

PROGRESS, EVENT, QUALITY = "progress", "event", "quality"

# object_height 를 subgoal 멤버로 쓸지. False: quality. True: progress+boundary=False.
OBJECT_HEIGHT_AS_SUBGOAL = False


# ===========================================================================
# 스키마
# ===========================================================================
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str            # progress | event | quality
    boundary: bool       # phase 를 자를 자격
    channel: str         # proposal Sec 7.1 채널
    doc: str


_OH_KIND = PROGRESS if OBJECT_HEIGHT_AS_SUBGOAL else QUALITY

FEATURES = [
    # ---- 경계 feature: 이 셋이 phase 를 만든다 (거리 2 + 그리퍼) --------------
    FeatureSpec("eef_object_dist",  PROGRESS, True,  "interaction",
                "그리퍼-물체 거리. plateau=접근 완료(approach->grasp 경계)"),
    FeatureSpec("object_goal_dist", PROGRESS, True,  "goal",
                "물체-목표 거리. plateau=배치 완료(transport->place 경계)"),
    FeatureSpec("gripper_open",     EVENT,    True,  "robot",
                "그리퍼 aperture proxy sum|qpos|. 닫힘=grasp 경계, 열림=place 경계"),

    # ---- subgoal 멤버 / 종료 조건, 하지만 경계는 아님 -------------------------
    FeatureSpec("contact",          EVENT,    False, "interaction",
                "그리퍼-물체 접촉/파지 flag. grasp/place 종료 조건의 일부"),
    FeatureSpec("grasp_align",      PROGRESS, False, "interaction",
                "eef-물체 상대 회전각. orientation 을 subgoal 로 실어 grasp 를 "
                "미정렬로 degrade 하게 함(Sec 8.1)"),

    # ---- object_height: 스위치로 quality<->progress ---------------------------
    FeatureSpec("object_height",    _OH_KIND, False, "object",
                "물체 들림 높이. 기본 quality(별도 lift phase 없음)"),

    # ---- quality(=LES) feature: 예측 대상, 경계도 subgoal 도 아님 -------------
    FeatureSpec("eef_speed",        QUALITY,  False, "robot",   "eef 속도"),
    FeatureSpec("object_speed",     QUALITY,  False, "object",  "물체 속도"),
    FeatureSpec("action_magnitude", QUALITY,  False, "robot",   "제어 노력"),
    FeatureSpec("eef_accel",        QUALITY,  False, "robot",   "eef 가속도"),
    FeatureSpec("eef_jerk",         QUALITY,  False, "robot",   "eef 저크"),
    FeatureSpec("object_slip",      QUALITY,  False, "interaction",
                "eef 프레임에서의 물체 이동(슬립/드리프트 proxy)"),
    FeatureSpec("eef_ang_speed",    QUALITY,  False, "robot",   "eef 각속도"),
    FeatureSpec("object_ang_speed", QUALITY,  False, "object",  "물체 각속도"),
]

NAMES = [f.name for f in FEATURES]
KINDS = [f.kind for f in FEATURES]
SPEC = {f.name: f for f in FEATURES}
BOUNDARY = [f.name for f in FEATURES if f.boundary]
SUBGOAL_ELIGIBLE = [f.name for f in FEATURES if f.kind != QUALITY]
N_FEATURES = len(FEATURES)
NON_DEFERRED = NAMES                        # 하위 호환 별칭 (모든 feature 를 직접 계산)


def index_of(name):
    return NAMES.index(name)


def boundary_mask():
    return np.array([f.boundary for f in FEATURES], dtype=bool)


def subgoal_eligible_mask():
    """quality 는 subgoal 에 못 들어간다: 큰 저크는 phase 실패가 아니라 비용."""
    return np.array([f.kind != QUALITY for f in FEATURES], dtype=bool)


# ===========================================================================
# 회전 / 운동학 helper
# ===========================================================================
def qnorm(q):
    q = np.asarray(q, float); n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0])


def canonicalize(quats):
    """부호 일관 quaternion 시퀀스 (q 와 -q 는 같은 회전)."""
    q = np.array(quats, float).copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def rel_angle(q1, q2):
    """상대 회전 q1->q2 의 각(rad, [0,pi]). 부호 뒤집힘에 안전."""
    if not _HAS_SCIPY:
        d = abs(float(np.dot(qnorm(q1), qnorm(q2))))
        return float(2 * np.arccos(min(1.0, d)))
    r = _R.from_quat(qnorm(q1)).inv() * _R.from_quat(qnorm(q2))
    return float(r.magnitude())


def angular_speed(quats, dt):
    q = canonicalize(quats); T = len(q); out = np.zeros(T)
    for t in range(1, T):
        out[t] = rel_angle(q[t - 1], q[t]) / dt
    return out


def smooth_positions(pos, window=9, poly=2):
    """(T,3) 위치를 미분 전에 Savgol 스무딩. phase_segment.trend_ratio 도 사용."""
    pos = np.asarray(pos, float); T = len(pos)
    if not _HAS_SCIPY or T < poly + 2:
        return pos.copy()
    w = window if window <= T else (T if T % 2 == 1 else T - 1)
    if w % 2 == 0:
        w -= 1
    if w <= poly:
        return pos.copy()
    return savgol_filter(pos, window_length=w, polyorder=poly, axis=0, mode="interp")


def eef_kinematics(eef, dt):
    """스무딩된 속도/가속도/저크 벡터(각각 (T,3)); 앞쪽 프레임 패딩."""
    eef = np.asarray(eef, float); T = len(eef)
    es = smooth_positions(eef)
    vel = np.zeros((T, 3)); acc = np.zeros((T, 3)); jrk = np.zeros((T, 3))
    if T > 1:
        vel[1:] = np.diff(es, axis=0) / dt; vel[0] = vel[1]
    if T > 2:
        acc[2:] = np.diff(vel[1:], axis=0) / dt; acc[:2] = acc[2]
    if T > 3:
        jrk[3:] = np.diff(acc[2:], axis=0) / dt; jrk[:3] = jrk[3]
    return vel, acc, jrk


# ===========================================================================
# action perturbation 유틸 (M3 / fcm.py 가 import)
# ===========================================================================
def compose_axis_angle(a1, a2):
    """rotvec(R(a1)·R(a2)) -- SO(3) 에서 a1+a2 의 올바른 대응. 선형 덧셈은 작은 각에서만
    1차 근사이므로 큰 lambda 회전 perturbation 은 제대로 합성해야 한다."""
    if not _HAS_SCIPY:
        return np.asarray(a1, float) + np.asarray(a2, float)
    r = _R.from_rotvec(np.asarray(a1, float)) * _R.from_rotvec(np.asarray(a2, float))
    return r.as_rotvec()


def perturb_action(a, delta, adim, proper_rotation=False, clip=True):
    """a + delta. 회전 차원(3..adim-2)은 옵션으로 제대로 합성. 컨트롤러 [-1,1] 박스로의
    클리핑은 기본 ON: 클리핑 안 하면 컨트롤러가 saturate 되어 실제 실행된 게 불명확해진다."""
    a = np.asarray(a, float).copy(); delta = np.asarray(delta, float)
    out = a + delta
    if proper_rotation and adim > 4:
        out[3:adim - 1] = compose_axis_angle(a[3:adim - 1], delta[3:adim - 1])
    return np.clip(out, -1.0, 1.0) if clip else out


def clip_fraction(a, delta, adim):
    """a+delta 가 action 박스 밖으로 나가는 정도(0 = 완전 feasible). 후보 방향의
    feasibility 가드로 쓴다."""
    raw = np.asarray(a, float) + np.asarray(delta, float)
    excess = np.maximum(np.abs(raw) - 1.0, 0.0)
    denom = np.abs(np.asarray(delta, float)).sum() + 1e-9
    return float(excess.sum() / denom)


# ===========================================================================
# feature 계산: frames -> (T, N_FEATURES) 행렬 (열 순서 NAMES)
# ===========================================================================
def compute_trajectory(eef, obj, grip, dt, eef_quat=None, obj_quat=None,
                       goal=None, obj0_z=None, actions=None, contact=None):
    """회전 채널은 quat 이 없으면 0(4-dim 호환). action_magnitude[t]=||actions[t-1]||."""
    eef = np.asarray(eef, float); obj = np.asarray(obj, float)
    grip = np.asarray(grip, float); T = len(eef)
    goal = obj[-1] if goal is None else np.asarray(goal, float)
    obj0_z = obj[0, 2] if obj0_z is None else float(obj0_z)

    eef_obj = np.linalg.norm(eef - obj, axis=1)
    obj_goal = np.linalg.norm(obj - goal[None, :], axis=1)
    obj_h = obj[:, 2] - obj0_z

    eef_speed = np.zeros(T); obj_speed = np.zeros(T)
    if T > 1:
        eef_speed[1:] = np.linalg.norm(np.diff(eef, axis=0), axis=1) / dt
        obj_speed[1:] = np.linalg.norm(np.diff(obj, axis=0), axis=1) / dt

    act_mag = np.zeros(T)
    if actions is not None and T > 1:
        an = np.linalg.norm(np.asarray(actions, float), axis=1)
        m = min(T - 1, len(an)); act_mag[1:1 + m] = an[:m]

    _, acc, jrk = eef_kinematics(eef, dt)
    eef_accel = np.linalg.norm(acc, axis=1); eef_jerk = np.linalg.norm(jrk, axis=1)

    rel = obj - eef
    slip = np.zeros(T)
    if T > 1:
        slip[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1)

    con = np.zeros(T) if contact is None else np.asarray(contact, float)

    if eef_quat is None or obj_quat is None:
        eef_as = np.zeros(T); obj_as = np.zeros(T); align = np.zeros(T)
    else:
        eef_as = angular_speed(eef_quat, dt); obj_as = angular_speed(obj_quat, dt)
        eq = canonicalize(eef_quat); oq = canonicalize(obj_quat)
        align = np.array([rel_angle(eq[t], oq[t]) for t in range(T)])

    col = {
        "eef_object_dist": eef_obj, "object_goal_dist": obj_goal,
        "gripper_open": grip, "contact": con, "grasp_align": align,
        "object_height": obj_h, "eef_speed": eef_speed, "object_speed": obj_speed,
        "action_magnitude": act_mag, "eef_accel": eef_accel, "eef_jerk": eef_jerk,
        "object_slip": slip, "eef_ang_speed": eef_as, "object_ang_speed": obj_as,
    }
    return np.stack([col[n] for n in NAMES], axis=1)


# ===========================================================================
# 자동 feature 프로파일러 (blueprint v5 Sec 5.3–5.7)
# ===========================================================================
# 사람이 정한 kind/boundary 는 그대로 둔다(= fcm.py 채점 게이트 + 검증된 5-phase 경로가
# 의존). 프로파일러는 "데이터가 보여주는 구조"를 별도 soft score 로 계산해
# feature_profiles.json 에 저장하고, 다운스트림은 이를 참고(가중/게이트)만 한다.
# 어떤 점수도 hard label 로 고정하지 않는다(Sec 5.5). feature 이름으로 규칙을 만들지
# 않고 실제 trajectory signal 만 본다(Sec 5.2). 순환 import 를 피하려고 phase_segment
# 를 import 하지 않고 필요한 numpy helper 는 여기서 자체 구현한다.
PROFILE_CONFIG = {
    "smooth_window": 21,    # trend/plateau 측정 전 Savgol 스무딩 창
    "noise_scale": 0.30,    # variation_score 소프트 스케일
    "n_boot": 24,           # bootstrap_stability 반복 수
    "resample_len": 64,     # cross-demo 비교용 공통 길이
    "eps": 1e-9,
}


def _minmax_local(x):
    x = np.asarray(x, float); lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _smooth1d(x, window=None):
    window = PROFILE_CONFIG["smooth_window"] if window is None else window
    return smooth_positions(np.asarray(x, float).reshape(-1, 1), window=window).ravel()


def variation_score(x):
    """이 신호가 (잡음이든 구조든) 얼마나 움직이나. 상수=0, 뚜렷한 변화≈1."""
    n = _minmax_local(x)
    return float(np.clip(np.std(n) / PROFILE_CONFIG["noise_scale"], 0.0, 1.0))


def trend_score(x):
    """|net| / total-variation, 스무딩 후. 1=완전 단조(progress), ~0=잡음/진동. TV 는
    운동보다 계측 잡음이 지배하므로 스무딩이 핵심(실제 progress~0.35 vs 순수 잡음~0.02)."""
    g = _smooth1d(x)
    if len(g) < 3:
        return 0.0
    tv = float(np.abs(np.diff(g)).sum())
    return float(np.clip(abs(g[-1] - g[0]) / (tv + PROFILE_CONFIG["eps"]), 0.0, 1.0))


def _two_means(v, iters=25):
    """1D 2-means -> (c0, c1). 데이터에서 나온 두 군집 중심."""
    v = np.asarray(v, float); lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo:
        return lo, hi
    c0, c1 = lo, hi
    for _ in range(iters):
        mid = 0.5 * (c0 + c1)
        left, right = v[v <= mid], v[v > mid]
        n0 = left.mean() if len(left) else c0
        n1 = right.mean() if len(right) else c1
        if np.isclose(n0, c0) and np.isclose(n1, c1):
            break
        c0, c1 = n0, n1
    return c0, c1


def transition_score(x):
    """이봉(near-binary / 모드 전이) 정도. 두 군집으로 갈리고 각 군집이 좁을수록 1.
    gripper/contact 같은 event 가 높고, 램프/잡음은 낮다."""
    v = np.asarray(x, float)
    if set(np.unique(v).tolist()).issubset({0.0, 1.0}):
        return 1.0 if (0.0 < v.mean() < 1.0) else 0.0
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo:                         # 상수 -> 전이 없음(빈 슬라이스 NaN 방지)
        return 0.0
    c0, c1 = _two_means(v)
    gap = abs(c1 - c0)
    mid = 0.5 * (c0 + c1)
    left, right = v[v <= mid], v[v > mid]
    if gap <= PROFILE_CONFIG["eps"] or left.size == 0 or right.size == 0:
        return 0.0                        # 단일 군집 -> 전이 아님
    spread = float(np.std(left) + np.std(right))
    return float(np.clip(gap / (gap + 2.0 * spread + PROFILE_CONFIG["eps"]), 0.0, 1.0))


def plateau_score(x):
    """큰 변화 뒤 정체(plateau)로 들어가는가. 도착형 progress(거리)가 높다. 전반 TV 대비
    후반 TV 가 작을수록 1. 애초에 큰 변화가 없으면(상수/잡음) 0."""
    g = _smooth1d(x); T = len(g)
    if T < 6:
        return 0.0
    half = T // 2
    tv_early = float(np.abs(np.diff(g[:half])).sum())
    tv_late = float(np.abs(np.diff(g[half:])).sum())
    rng = float(np.max(g) - np.min(g)) + PROFILE_CONFIG["eps"]
    if abs(g[-1] - g[0]) / rng < 0.3:        # 큰 변화가 없으면 plateau 아님
        return 0.0
    return float(np.clip(1.0 - tv_late / (tv_early + PROFILE_CONFIG["eps"]), 0.0, 1.0))


def noise_ratio(x):
    """구조 대비 잡음. variation 은 있는데 trend 가 낮으면(=떨림) 높다."""
    return float(np.clip(variation_score(x) * (1.0 - trend_score(x)), 0.0, 1.0))


def _structural_from(sc):
    """이 feature 가 (어느 phase 에선가) 이용 가능한 시간 구조를 갖는가. progress(trend)/
    도착(plateau)/모드전이(transition) 중 가장 센 것에서 잡음을 뺀다."""
    s = max(sc["trend_score"], sc["plateau_score"], sc["transition_score"])
    return float(np.clip(s - 0.5 * sc["noise_ratio"], 0.0, 1.0))


def _boundary_from(sc):
    """phase 경계를 만들 만한가 = 도착(plateau) 또는 모드전이(transition)."""
    return float(max(sc["plateau_score"], sc["transition_score"]))


def raw_scores(x):
    """한 신호(1D)의 phase-무관 soft score 묶음. profile_feature 와 selftest 가 공유."""
    sc = {
        "variation_score": variation_score(x),
        "trend_score": trend_score(x),
        "transition_score": transition_score(x),
        "plateau_score": plateau_score(x),
    }
    sc["noise_ratio"] = noise_ratio(x)
    sc["structural_score"] = _structural_from(sc)
    sc["boundary_score"] = _boundary_from(sc)
    return sc


def _resample(x, n=None):
    n = PROFILE_CONFIG["resample_len"] if n is None else n
    x = np.asarray(x, float); T = len(x)
    if T == n or T < 2:
        return x if T == n else np.full(n, float(x[0]) if T else 0.0)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, T), x)


def cross_demo_consistency(signals):
    """여러 데모에서 phase-상대 위치의 신호 형태가 반복되는가. 각 데모를 공통 길이로
    리샘플해 쌍별 상관의 평균. 데모가 1개면 (0, debug_only=True)."""
    sigs = [np.asarray(s, float) for s in signals if len(np.asarray(s)) >= 3]
    if len(sigs) < 2:
        return 0.0, True
    R = np.stack([_minmax_local(_resample(s)) for s in sigs], axis=0)
    corr = []
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            a, b = R[i] - R[i].mean(), R[j] - R[j].mean()
            d = float(np.linalg.norm(a) * np.linalg.norm(b)) + PROFILE_CONFIG["eps"]
            corr.append(float(np.dot(a, b) / d))
    return float(np.clip(np.mean(corr), 0.0, 1.0)), False


def bootstrap_stability(x, rng):
    """연속 부분구간을 반복 샘플했을 때 structural_score 가 유지되나. (안정도, 불확실도)."""
    x = np.asarray(x, float); T = len(x)
    if T < 8:
        return 0.0, 1.0
    vals = []
    for _ in range(PROFILE_CONFIG["n_boot"]):
        k = int(rng.integers(T // 2, T))
        start = int(rng.integers(0, T - k + 1))
        vals.append(_structural_from(raw_scores(x[start:start + k])))
    vals = np.asarray(vals)
    return float(np.clip(1.0 - 2.0 * vals.std(), 0.0, 1.0)), float(vals.std())


def profile_feature(name, per_demo_signals, rng):
    """한 feature 를 여러 데모에서 프로파일. per-demo 점수를 평균하고 cross-demo /
    bootstrap 를 더한다. hard label 없이 soft score 만 반환(Sec 5.5). kind_hint 는
    참고용일 뿐 정답이 아니다."""
    keys = ["variation_score", "trend_score", "transition_score",
            "plateau_score", "noise_ratio", "structural_score", "boundary_score"]
    acc = {k: [] for k in keys}
    boots, uncs = [], []
    for x in per_demo_signals:
        sc = raw_scores(x)
        for k in keys:
            acc[k].append(sc[k])
        bs, unc = bootstrap_stability(x, rng)
        boots.append(bs); uncs.append(unc)
    out = {k: float(np.mean(acc[k])) if acc[k] else 0.0 for k in keys}
    cdc, debug_only = cross_demo_consistency(per_demo_signals)
    out["cross_demo_consistency"] = cdc
    out["bootstrap_stability"] = float(np.mean(boots)) if boots else 0.0
    out["structural_uncertainty"] = float(np.mean(uncs)) if uncs else 1.0
    out["confidence"] = float(np.clip(
        out["variation_score"] * (0.5 + 0.5 * out["structural_score"])
        * out["bootstrap_stability"], 0.0, 1.0))
    out["debug_only"] = bool(debug_only)
    out["name"] = name
    out["kind_hint"] = SPEC[name].kind if name in SPEC else "unknown"
    return out


def profile_all(feature_mats, names=None, seed=42):
    """feature_mats: 데모별 raw feature 행렬 리스트 [(T_i, N), ...]. 열 순서는 names(기본
    NAMES). 반환 {name: profile dict}. 단일 데모면 cross_demo_consistency=0, debug_only."""
    names = list(NAMES) if names is None else list(names)
    rng = np.random.default_rng(seed)
    return {n: profile_feature(n, [np.asarray(F, float)[:, i] for F in feature_mats], rng)
            for i, n in enumerate(names)}


# ===========================================================================
# robust normalization (train-only 통계; blueprint Sec 5.4)
# ===========================================================================
def robust_normalization_stats(feature_mats, names=None):
    """train 데모들을 이어붙여 feature 별 center=median, scale=IQR(>0) 계산. IQR 이 0 이면
    MAD 로 대체. held-out 에는 이 통계만 적용한다(train-only)."""
    names = list(NAMES) if names is None else list(names)
    allF = np.concatenate([np.asarray(F, float) for F in feature_mats], axis=0)
    stats = {}
    for i, n in enumerate(names):
        col = allF[:, i]
        med = float(np.median(col))
        q75, q25 = np.percentile(col, [75, 25])
        iqr = float(q75 - q25)
        if iqr <= 1e-9:
            iqr = float(1.4826 * np.median(np.abs(col - med)))   # MAD fallback
        stats[n] = {"center": med, "scale": float(max(iqr, 1e-6))}
    return stats


def apply_normalization(F, stats, names=None):
    """center/scale 로 (x-center)/scale. 원본과 정규화본을 둘 다 저장하기 위한 변환."""
    names = list(NAMES) if names is None else list(names)
    F = np.asarray(F, float); out = np.empty_like(F)
    for i, n in enumerate(names):
        out[:, i] = (F[:, i] - stats[n]["center"]) / stats[n]["scale"]
    return out


# ===========================================================================
# artifact I/O — raw_demo_XXX.npz -> features_XXX.npz + profiles  (Sec 5.5–5.6)
# ===========================================================================
RAW_SCHEMA_VERSION = "raw-1"
FEATURES_SCHEMA_VERSION = "features-1"

# collect_demo.py extract 가 써야 할 raw npz 스키마(blueprint Sec 4.2/4.4). feature 계산에
# 실제로 필요한 것만 필수로 둔다. 나머지(qpos/qvel/states)는 있으면 통과, 없어도 무방.
RAW_REQUIRED_KEYS = ["eef_pos", "object_pos", "gripper_aperture"]
RAW_OPTIONAL_KEYS = ["time", "actions", "eef_quat", "object_quat", "contact",
                     "goal_context", "goal", "qpos", "qvel", "states", "dt"]


def read_raw_npz(path):
    """raw_demo_XXX.npz 를 읽어 compute_trajectory 인자 dict 로 변환. 필수 키가 없으면
    조용히 넘기지 않고 명확히 오류(Sec 11.5 silent fallback 금지)."""
    d = np.load(path, allow_pickle=True)
    for k in RAW_REQUIRED_KEYS:
        if k not in d:
            raise KeyError(f"{path}: raw npz 에 필수 키 '{k}' 없음 "
                           f"(collect_demo.py extract 스키마 Sec 4.2 확인). "
                           f"필수={RAW_REQUIRED_KEYS}")
    eef = np.asarray(d["eef_pos"], float)
    obj = np.asarray(d["object_pos"], float)
    grip = np.asarray(d["gripper_aperture"], float)
    contact = np.asarray(d["contact"], float) if "contact" in d else None
    eef_q = np.asarray(d["eef_quat"], float) if "eef_quat" in d else None
    obj_q = np.asarray(d["object_quat"], float) if "object_quat" in d else None
    actions = np.asarray(d["actions"], float) if "actions" in d else None
    if "time" in d and len(np.asarray(d["time"])) > 1:
        dt = float(np.median(np.diff(np.asarray(d["time"], float))))
    elif "dt" in d:
        dt = float(d["dt"])
    else:
        dt = 0.05
    goal = None
    if "goal_context" in d:
        gc = d["goal_context"]
        gc = gc.item() if hasattr(gc, "item") and gc.dtype == object else gc
        if isinstance(gc, dict) and "target_pos" in gc:
            goal = np.asarray(gc["target_pos"], float)
    if goal is None and "goal" in d and len(np.asarray(d["goal"])) == 3:
        goal = np.asarray(d["goal"], float)
    return dict(eef=eef, obj=obj, grip=grip, contact=contact, eef_q=eef_q,
                obj_q=obj_q, actions=actions, dt=dt, goal=goal, path=path)


def _demo_tag(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return base[4:] if base.startswith("raw_") else base


def run_compute(input_dir, output_dir, seed=42):
    """artifacts/raw/*.npz -> features_XXX.npz + normalization_stats.json +
    feature_profiles.json + feature_manifest.json. 원본/정규화 feature 를 둘 다 저장."""
    paths = sorted(glob.glob(os.path.join(input_dir, "raw_demo_*.npz"))) \
        or sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    if not paths:
        raise SystemExit(f"{input_dir} 아래 raw npz 없음. "
                         f"먼저 `python collect_demo.py extract` 로 raw 를 만들어야 함.")
    os.makedirs(output_dir, exist_ok=True)
    raws, mats, tags = [], [], []
    for p in paths:
        raw = read_raw_npz(p)
        F = compute_trajectory(
            raw["eef"], raw["obj"], raw["grip"], raw["dt"], raw["eef_q"], raw["obj_q"],
            goal=raw["goal"], obj0_z=float(raw["obj"][0, 2]),
            actions=raw["actions"], contact=raw["contact"])
        if not np.isfinite(F).all():
            raise ValueError(f"{p}: 계산된 feature 에 NaN/Inf")
        raws.append(raw); mats.append(F); tags.append(_demo_tag(p))

    stats = robust_normalization_stats(mats)
    prof = profile_all(mats, seed=seed)
    goal_used = all(r["goal"] is not None for r in raws)

    for tag, F, raw in zip(tags, mats, raws):
        Fn = apply_normalization(F, stats)
        np.savez(os.path.join(output_dir, f"features_{tag}.npz"),
                 features_raw=F, features_norm=Fn,
                 feat_names=np.array(NAMES, dtype=object), dt=raw["dt"],
                 goal=(raw["goal"] if raw["goal"] is not None else np.array([])),
                 goal_from_context=bool(raw["goal"] is not None),
                 schema_version=FEATURES_SCHEMA_VERSION)

    with open(os.path.join(output_dir, "normalization_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(output_dir, "feature_profiles.json"), "w") as f:
        json.dump(prof, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "feature_manifest.json"), "w") as f:
        json.dump({"schema_version": FEATURES_SCHEMA_VERSION,
                   "feature_name_order": NAMES, "n_features": N_FEATURES,
                   "n_demos": len(paths), "kinds": KINDS, "boundary": BOUNDARY,
                   "seed": seed, "goal_from_context": goal_used}, f, indent=2)

    print(f"[Stage 4-5 {'PASS' if goal_used else 'WARN'}]")
    print(f"Processed demos: {len(paths)}")
    print(f"Artifacts: {output_dir}/ (features_*.npz, normalization_stats.json, "
          f"feature_profiles.json, feature_manifest.json)")
    if not goal_used:
        print("[warn] 일부 demo 에 goal_context 없음 -> object_goal_dist 가 데모 최종위치 "
              "fallback 을 씀. collect_demo extract 의 goal_context 확인 권장.")
    print(f"\n{'feature':<18}{'struct':>7}{'bound':>7}{'trend':>7}{'trans':>7}"
          f"{'plat':>7}{'noise':>7}{'xdemo':>7}")
    for n in NAMES:
        p = prof[n]
        print(f"{n:<18}{p['structural_score']:>7.2f}{p['boundary_score']:>7.2f}"
              f"{p['trend_score']:>7.2f}{p['transition_score']:>7.2f}"
              f"{p['plateau_score']:>7.2f}{p['noise_ratio']:>7.2f}"
              f"{p['cross_demo_consistency']:>7.2f}")
    print(f"\nNext: python phase_segment.py all --features {output_dir} "
          f"--output artifacts/segmentation")
    return prof


def run_profile_only(features_dir, seed=42):
    """이미 만든 features_*.npz 로 프로파일만 재계산해 feature_profiles.json 갱신."""
    paths = sorted(glob.glob(os.path.join(features_dir, "features_*.npz")))
    if not paths:
        raise SystemExit(f"{features_dir} 아래 features_*.npz 없음. 먼저 compute 실행.")
    mats = [np.asarray(np.load(p, allow_pickle=True)["features_raw"], float) for p in paths]
    prof = profile_all(mats, seed=seed)
    with open(os.path.join(features_dir, "feature_profiles.json"), "w") as f:
        json.dump(prof, f, indent=2, ensure_ascii=False)
    print(f"[ok] {len(paths)} demos 재프로파일 -> {features_dir}/feature_profiles.json")
    return prof


# ===========================================================================
# selftest
# ===========================================================================
def run_selftest():
    print("=== feature_select SELFTEST ===")
    ok = True
    print(f"{N_FEATURES} features   (OBJECT_HEIGHT_AS_SUBGOAL={OBJECT_HEIGHT_AS_SUBGOAL})\n")
    print(f"{'name':<18}{'kind':<10}{'boundary':<10}subgoal-eligible")
    for f in FEATURES:
        print(f"{f.name:<18}{f.kind:<10}{str(f.boundary):<10}{f.kind != QUALITY}")

    # 스키마 불변식 — image 2 의 5단계를 만드는 경계 집합
    if BOUNDARY != ["eef_object_dist", "object_goal_dist", "gripper_open"]:
        ok = False; print(f"[FAIL] 경계는 거리 2 + 그리퍼 여야 함. got {BOUNDARY}")
    if not (SPEC["gripper_open"].kind == EVENT and SPEC["gripper_open"].boundary):
        ok = False; print("[FAIL] gripper_open 은 event 이고 boundary=True 여야 함")
    if SPEC["contact"].kind != EVENT or SPEC["contact"].boundary:
        ok = False; print("[FAIL] contact 은 event 이고 boundary=False 여야 함")
    if SPEC["grasp_align"].kind != PROGRESS or SPEC["grasp_align"].boundary:
        ok = False; print("[FAIL] grasp_align 은 progress 이고 boundary=False 여야 함")
    for f in FEATURES:
        if f.kind == QUALITY and f.boundary:
            ok = False; print(f"[FAIL] quality {f.name} 은 경계가 될 수 없음")
        if "waypoint" in f.name:
            ok = False; print(f"[FAIL] {f.name}: waypoint feature 는 제거됨")

    print(f"\n경계 -> phase:      {BOUNDARY}")
    print(f"subgoal-eligible:   {SUBGOAL_ELIGIBLE}")

    # 계산 확인
    T, dt = 120, 0.05
    eef = np.zeros((T, 3)); obj = np.zeros((T, 3))
    eef[:, 0] = np.linspace(0.0, 0.5, T)
    eef[:, 2] = 1.0 - 0.15 * np.sin(np.linspace(0, np.pi, T))
    obj[:60] = [0.25, 0.0, 0.85]
    obj[60:, 0] = np.linspace(0.25, 0.5, T - 60); obj[60:, 2] = 0.85
    grip = np.zeros(T); grip[60:110] = 1.0
    con = np.zeros(T); con[60:110] = 1.0
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (T, 1))
    acts = np.zeros((T - 1, 7))

    F = compute_trajectory(eef, obj, grip, dt, quat, quat, goal=obj[-1],
                           actions=acts, contact=con)
    if F.shape != (T, N_FEATURES):
        ok = False; print(f"[FAIL] shape {F.shape} != {(T, N_FEATURES)}")
    if not np.allclose(F[:, index_of("contact")], con):
        ok = False; print("[FAIL] contact 열 미연결")
    print(f"\ncompute_trajectory -> {F.shape}")

    # perturb / clip 유틸 (fcm.py 가 import) 확인
    cf = clip_fraction(np.array([0.9, 0, 0, 0, 0, 0, 0]),
                       np.array([0.5, 0, 0, 0, 0, 0, 0]), 7)
    if not (cf > 0):
        ok = False; print("[FAIL] clip_fraction(0.9+0.5) 은 >0 이어야 함")
    pa = perturb_action(np.zeros(7), np.array([2.0, 0, 0, 0, 0, 0, 0]), 7)
    if not np.all(np.abs(pa) <= 1.0 + 1e-9):
        ok = False; print("[FAIL] perturb_action 이 [-1,1] 로 클리핑 안 함")
    print(f"clip_fraction(0.9+0.5)={cf:.2f}, perturb_action clip OK")

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def run_profiler_selftest():
    """합성 신호로 automatic profiler 를 검증(blueprint Sec 5.7). profiler 가 정해진
    형태(단조/도착/이진/잡음 등)를 점수로 구분하는지 확인. robosuite 불필요."""
    print("\n=== feature_select PROFILER SELFTEST (합성; robosuite 불필요) ===")
    rng = np.random.default_rng(0)
    T = 200
    t = np.linspace(0, 1, T)
    sigs = {
        "monotonic_decrease+plateau":
            np.concatenate([np.linspace(1, 0, 120), np.zeros(80)]) + rng.normal(0, 0.01, T),
        "monotonic_increase+plateau":
            np.concatenate([np.linspace(0, 1, 120), np.ones(80)]) + rng.normal(0, 0.01, T),
        "binary_0to1":         (t > 0.5).astype(float),
        "binary_1to0":         (t < 0.5).astype(float),
        "multiple_noisy_trans": (np.sin(2 * np.pi * 6 * t) > 0).astype(float),
        "pure_gaussian_noise": rng.normal(0, 1, T),
        "sinusoidal":          np.sin(2 * np.pi * 2 * t),
        "early_outlier":       np.r_[5.0, np.zeros(T - 1)] + rng.normal(0, 0.01, T),
        "pause_segment":       np.concatenate([np.linspace(0, 1, 80), np.ones(40),
                                               np.linspace(1, 2, 80)]),
        "sidetrack_recovery":  np.concatenate([np.linspace(0, 1, 90), np.linspace(1, 0.7, 20),
                                               np.linspace(0.7, 1.5, 90)]),
    }
    scores = {k: raw_scores(v) for k, v in sigs.items()}

    print(f"{'signal':<26}{'var':>6}{'trend':>7}{'trans':>7}{'plat':>7}"
          f"{'noise':>7}{'struct':>7}{'bound':>7}")
    for k, s in scores.items():
        print(f"{k:<26}{s['variation_score']:>6.2f}{s['trend_score']:>7.2f}"
              f"{s['transition_score']:>7.2f}{s['plateau_score']:>7.2f}"
              f"{s['noise_ratio']:>7.2f}{s['structural_score']:>7.2f}"
              f"{s['boundary_score']:>7.2f}")

    ok = True

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            ok = False; print(f"[FAIL] {msg}")

    # 1) 순수 잡음은 구조가 낮고 잡음비가 높다
    chk(scores["pure_gaussian_noise"]["structural_score"] < 0.35,
        "pure noise 의 structural_score 가 낮아야")
    chk(scores["pure_gaussian_noise"]["noise_ratio"] > 0.4,
        "pure noise 의 noise_ratio 가 높아야")
    # 2) 단조 progress 는 trend/plateau 를 잡는다
    chk(scores["monotonic_decrease+plateau"]["trend_score"] > 0.5,
        "monotonic 의 trend_score 가 높아야")
    chk(scores["monotonic_decrease+plateau"]["plateau_score"] > 0.5,
        "도착형의 plateau_score 가 높아야")
    # 3) 이진 event 는 transition/구조를 잡는다
    chk(scores["binary_0to1"]["transition_score"] > 0.7,
        "binary 의 transition_score 가 높아야")
    chk(scores["binary_0to1"]["structural_score"] > 0.6,
        "binary 는 구조가 있어야")
    # 4) 구조 신호가 잡음보다 structural 높다
    chk(scores["monotonic_increase+plateau"]["structural_score"]
        > scores["pure_gaussian_noise"]["structural_score"],
        "구조 신호가 잡음보다 structural_score 가 높아야")
    # 5) 왕복 sine 은 net~0 이라 trend 가 낮다(단조가 아님)
    chk(scores["sinusoidal"]["trend_score"] < 0.3,
        "sine 은 net~0 이라 trend 가 낮아야")

    # 6) cross-demo: 같은 형태 3개가 무관 3개보다 일관성 높음
    same = [sigs["monotonic_decrease+plateau"],
            sigs["monotonic_decrease+plateau"] + rng.normal(0, 0.02, T),
            sigs["monotonic_decrease+plateau"] + rng.normal(0, 0.02, T)]
    mixed = [sigs["monotonic_decrease+plateau"], sigs["binary_0to1"],
             sigs["pure_gaussian_noise"]]
    cdc_same, dbg_same = cross_demo_consistency(same)
    cdc_mixed, _ = cross_demo_consistency(mixed)
    cdc_single, dbg_single = cross_demo_consistency([sigs["binary_0to1"]])
    print(f"\ncross_demo_consistency  same-shape={cdc_same:.2f}  mixed={cdc_mixed:.2f}  "
          f"single={cdc_single:.2f}(debug_only={dbg_single})")
    chk(cdc_same > cdc_mixed, "같은 형태의 cross-demo consistency 가 더 높아야")
    chk(cdc_same > 0.8, "같은 형태는 cross-demo consistency ~1")
    chk(dbg_single and cdc_single == 0.0, "단일 데모는 cross-demo=0 & debug_only")

    # 7) 상수(0/1 아님) feature 는 NaN 을 만들면 안 됨 (transition_score 빈-슬라이스 가드)
    for cval in (0.0, 0.5, 0.85, 1.0, -3.0):
        cs = raw_scores(np.full(T, cval))
        finite = all(np.isfinite(x) for x in cs.values())
        chk(finite, f"상수 feature(={cval}) 점수에 NaN/Inf: {cs}")
        chk(cs["transition_score"] == 0.0, f"상수 feature(={cval}) 는 transition=0 이어야")
    cf_prof = profile_feature("const_probe", [np.full(T, 0.85)] * 3,
                              np.random.default_rng(2))
    chk(all(np.isfinite(v) for k, v in cf_prof.items()
            if isinstance(v, (int, float))),
        "상수 feature profile_feature 에 NaN/Inf (feature_profiles.json 오염)")

    # 8) profile_feature 종단 확인 (모든 키 존재, 유한값)
    pf = profile_feature("eef_object_dist", same, np.random.default_rng(1))
    need = {"structural_score", "boundary_score", "confidence", "bootstrap_stability",
            "structural_uncertainty", "cross_demo_consistency", "debug_only"}
    chk(need.issubset(pf.keys()), f"profile_feature 키 누락: {need - set(pf.keys())}")
    chk(all(np.isfinite(v) for k, v in pf.items()
            if isinstance(v, (int, float))), "profile_feature 에 비유한값")

    print(f"\n[profiler selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="feature 스키마 + automatic profiler + artifact I/O.")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("all", "compute"):
        p = sub.add_parser(name, help="raw -> features + profiles")
        p.add_argument("--input", default="artifacts/raw")
        p.add_argument("--output", default="artifacts/features")
        p.add_argument("--seed", type=int, default=42)
    pp = sub.add_parser("profile", help="features -> profiles 재계산")
    pp.add_argument("--input", default="artifacts/features")
    pp.add_argument("--seed", type=int, default=42)
    sub.add_parser("self-test")
    sub.add_parser("selftest")
    ap.add_argument("--selftest", action="store_true", help="하위 호환 플래그")
    args = ap.parse_args()

    if args.selftest or args.cmd in ("self-test", "selftest"):
        ok = run_selftest() and run_profiler_selftest()
        raise SystemExit(0 if ok else 1)
    if args.cmd in ("all", "compute"):
        run_compute(args.input, args.output, seed=args.seed)
    elif args.cmd == "profile":
        run_profile_only(args.input, seed=args.seed)
    else:
        print(f"{N_FEATURES} features: {NAMES}")
        print(f"boundary: {BOUNDARY}")


if __name__ == "__main__":
    main()