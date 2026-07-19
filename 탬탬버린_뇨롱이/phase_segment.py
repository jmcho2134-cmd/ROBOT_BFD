#!/usr/bin/env python
"""
phase_segment.py — M1: phase 세그먼트 + phase 별 subgoal set.
================================================================================

스키마는 feature_select.py 에서만 읽고 여기서는 아무 feature 도 정의하지 않는다.
세그먼터는 "어느 feature 가 approach 냐"를 배우지 않는다. 각 열의 KIND 와 경계
생성 자격(BOUNDARY)만 본다.

    progress + boundary -> high-water-mark 의 plateau 가 경계 (Kneedle knee, 임계값
                           없음. 누적 최댓값이 sidetrack/regression 을 안 보이게 해서,
                           오직 PROGRESS 만 phase 를 닫음 -> 미분류 비효율은 흡수됨)
    event    + boundary -> 모드 전이가 경계 (2-means split, 분할점은 데이터에서)
    quality             -> 여기선 완전히 무시

image 2 의 5단계는 이 조합에서 나온다: 거리 두 개(도착) + 그리퍼(닫힘/열림)가 경계라서
approach / grasp / transport / place / retreat 로 저절로 갈린다. phase 개수·이름·순서·
임계값은 하드코딩하지 않는다.

phase 는 feature 하나가 아니다. approach 는 "eef-물체 거리가 줄고, 그 동안 물체는
가만히 있고, 그리퍼는 안 움직이고, 아무 것도 안 닿는" 패턴 전체다. 그래서 subgoal 을
데모에서 읽어 역할을 매긴다:
    main   : 그 phase 를 닫은 feature (그 phase 가 무엇에 관한 것인가)
    change : 데모가 몰아간 feature -> degrade = 반대로 (부호 포함)
    hold   : 데모가 ~일정하게 유지한 feature -> degrade = 흔들기
    free   : 진동(quality 성격) -> subgoal 아님

near-binary 인 그리퍼가 grasp 를 닫아도, 같은 phase 가 contact 를 change,
grasp_align / eef_object_dist 를 hold 하므로 그 연속 feature 로 degrade(미정렬/접촉
상실, Sec 8.1) 가 가능하다. degrade 채점 자체는 fcm.py 가 이 set 전체로 한다.

그리퍼 라벨은 ORDER 로 (1번째 전이=닫힘/grasp, 2번째=열림/place). rise/fall 로 하면
Panda 는 닫힐 때 sum|qpos| 가 RISE 라 극성 의존이 되어 grasp/place 가 뒤집힌다.

waypoint feature 는 없다: 데모 자기 waypoint 와의 거리는 Sec 9.3 similarity trap.

재사용: segment_features() 는 순수 numpy 이고 M5 rollout 에도 같은 호출을 쓴다
(reward 가 z_t 를 입력받으므로 새 궤적도 같은 방식으로 나뉘어야 함, Sec 6.2).

    python phase_segment.py --selftest              # robosuite 불필요
    python phase_segment.py all --features artifacts/features --output artifacts/segmentation
"""

import argparse
import json
import os
from glob import glob

import numpy as np

import feature_select as fs


# ===========================================================================
# SECTION 1 — 세그먼터 (순수 numpy; M5 rollout 재사용)
# ===========================================================================
def _minmax(x):
    x = np.asarray(x, float); lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _is_decreasing(f, edge_frac=0.1):
    f = np.asarray(f, float); e = max(1, int(edge_frac * len(f)))
    return f[-e:].mean() <= f[:e].mean()


def normalized_progress(f, edge_frac=0.1):
    """progress feature 를 [0,1] 로, 1=가장 완성. 방향은 net 추세로 자동 판별."""
    n = _minmax(f)
    return (1.0 - n) if _is_decreasing(f, edge_frac) else n


def high_water_mark(p):
    """누적 최댓값 -> 단조. regression(sidetrack)은 사라진다."""
    return np.maximum.accumulate(np.asarray(p, float))


def trend_ratio(f, window=21):
    """|net 변화| / total variation, 스무딩한 신호 위에서. 1=완전 단조, ~0=순수 잡음.
    "여기서 실제로 진행했나 vs 떨렸을 뿐인가"를 가른다. TV 는 운동보다 계측 잡음이
    지배하므로 스무딩이 핵심(실제 progress ~0.35 vs 순수 잡음 ~0.016)."""
    f = np.asarray(f, float)
    if len(f) < 5:
        return 0.0
    g = fs.smooth_positions(f.reshape(-1, 1), window=window).ravel()
    tv = float(np.abs(np.diff(g)).sum())
    return abs(float(g[-1] - g[0])) / (tv + 1e-12)


def plateau_knee(hwm):
    """Kneedle: 오목 증가 곡선에서 코드(chord) 위로 가장 먼 점. 임계값 없음."""
    hwm = np.asarray(hwm, float); T = len(hwm)
    if T < 3:
        return T - 1
    return int(np.argmax(_minmax(hwm) - np.linspace(0.0, 1.0, T)))


def two_means_threshold(v, iters=25):
    """1D 2-means -> (threshold, high_is_upper). 데이터에서 나온 분할점."""
    v = np.asarray(v, float); lo, hi = np.min(v), np.max(v)
    if hi <= lo:
        return lo, True
    c0, c1 = lo, hi
    for _ in range(iters):
        mid = 0.5 * (c0 + c1)
        left, right = v[v <= mid], v[v > mid]
        n0 = left.mean() if len(left) else c0
        n1 = right.mean() if len(right) else c1
        if np.isclose(n0, c0) and np.isclose(n1, c1):
            break
        c0, c1 = n0, n1
    return 0.5 * (c0 + c1), (c1 >= c0)


def event_transitions(f, min_run=3):
    """2-means 로 이진화 후 각 flip 을 (time, +1 rise / -1 fall) 로. min_run 은 그리퍼
    지터가 가짜 phase 를 만들지 않게 디바운스."""
    f = np.asarray(f, float)
    if set(np.unique(f).tolist()).issubset({0.0, 1.0}):
        b = f.astype(int)
    else:
        thr, high_upper = two_means_threshold(f)
        b = (f > thr).astype(int)
        if not high_upper:
            b = 1 - b
    out = b.copy(); i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if (j - i) < min_run and 0 < i < len(out):
            out[i:j] = out[i - 1]
        i = j
    idx = np.where(np.diff(out) != 0)[0]
    return [(int(i + 1), int(out[i + 1] - out[i])) for i in idx]


def _tail_subgoal(F, a, b, names, so_far, min_trend=0.15):
    """어떤 사건도 닫지 않은 구간(retreat)의 subgoal: 이 구간에서 데모가 아직 가장 세게
    몰아가는 boundary-eligible feature. sign = -sign(net): 데모가 그리 몰았으니 degrade 는
    반대. retreat 에선 (eef_object_dist, -1) = "멀어져야 하는데 안 멀어짐" 이 되어 연속이고
    잘 degrade 된다. retreat 는 degrade 해도 task 가 실패할 수 없는 유일한 phase =
    pure-efficiency family (BfD 가 딱 필요로 하는 것)."""
    best, best_tr = None, min_trend
    for i, n in enumerate(names):
        sp = fs.SPEC.get(n)
        if sp is None or not sp.boundary:
            continue
        seg = F[a:b, i]
        tr = trend_ratio(seg)
        if tr > best_tr:
            net = float(seg[-1] - seg[0])
            if net != 0:
                best, best_tr = (n, -float(np.sign(net))), tr
    if best is not None:
        return best
    return so_far[-1] if so_far else (names[0], +1.0)


def _profile_score(profiles, name):
    """이 feature 의 경계 신뢰도 = max(structural_score, boundary_score). structural 은
    전역 단조/구조라 phase-local feature(예: eef_object_dist: approach 에서만 급감)를
    과소평가하므로, boundary-shape(plateau/transition) 신뢰도와 함께 max 로 본다."""
    p = profiles.get(name) if profiles else None
    if not p:
        return 1.0
    return max(float(p.get("structural_score", 1.0)),
               float(p.get("boundary_score", 1.0)))


def _profile_gate_ok(profiles, name, min_score):
    """profiles 가 주어지면, boundary-eligible feature 라도 이 데모에서 구조/경계 신뢰도가
    둘 다 약하면(max(structural, boundary) < min_score) 경계 생성에서 제외. 즉 사실상
    죽은(상수/잡음) feature 만 걸러지고 실제 경계 feature 는 보호된다. profiles=None 이면
    항상 통과라 기존 5-phase 경로는 완전히 동일(전환형: 참고만, 정답은 스키마)."""
    if profiles is None:
        return True
    return _profile_score(profiles, name) >= min_score


def segment_features(F, names=None, min_seg_frac=0.03, min_trend=0.15,
                     profiles=None, min_score=0.15):
    """feature-불가지 세그먼트. M1 과 M5 가 공유하는 진입점.

    profiles(옵션): feature_select.profile_all() 이 만든 {name: {structural_score,...}}.
    주면 boundary-eligible feature 를 이 데모에서의 구조 점수로 추가 게이팅한다(참고용:
    스키마의 boundary=True 를 정답으로 유지하되, 데이터가 구조 없다고 말하는 feature 의
    가짜 경계만 제거). profiles=None 이면 동작은 종전과 100% 동일하다.

    반환 dict:
      z                 (T,) 시점별 phase 번호
      bounds            경계 시점들
      labels            각 phase 를 닫은 것
      subgoal_per_phase [(feature_name, degrade_sign), ...] phase 당 하나(main)
      events            진단용
      profile_gated     profiles 게이트로 제외된 (name, structural_score) (진단용)
    """
    F = np.asarray(F, float); T, k = F.shape
    names = list(names) if names is not None else list(fs.NON_DEFERRED)
    if len(names) != k:
        raise ValueError(f"F 는 {k} 열인데 이름은 {len(names)} 개")
    min_seg = max(2, int(min_seg_frac * T))

    events = []                                   # (t, name, type, dir, col_idx)
    profile_gated = []

    # -- progress + boundary: high-water-mark 의 plateau --------------------
    for i, n in enumerate(names):
        sp = fs.SPEC.get(n)
        if sp is None or not sp.boundary or sp.kind != fs.PROGRESS:
            continue
        if trend_ratio(F[:, i]) < min_trend:
            continue          # 여기선 떨렸을 뿐 -> subgoal 아님
        if not _profile_gate_ok(profiles, n, min_score):
            profile_gated.append((n, round(_profile_score(profiles, n), 3)))
            continue          # 데이터가 "이 데모엔 구조 없음"이라고 말함 -> 경계 스킵
        hwm = high_water_mark(normalized_progress(F[:, i]))
        events.append((plateau_knee(hwm), n, "plateau", 0, i))

    # -- event + boundary: 모드 전이 --------------------------------------
    for i, n in enumerate(names):
        sp = fs.SPEC.get(n)
        if sp is None or not sp.boundary or sp.kind != fs.EVENT:
            continue
        if not _profile_gate_ok(profiles, n, min_score):
            profile_gated.append((n, round(_profile_score(profiles, n), 3)))
            continue
        for t, d in event_transitions(F[:, i]):
            events.append((t, n, "transition", d, i))

    events.sort(key=lambda e: e[0])

    # event feature 의 engage/release 를 ORDER 로 (극성 무관).
    ev_order, seen = {}, {}
    for e in events:
        if e[2] == "transition":
            c = seen.get(e[1], 0); ev_order[(e[1], e[0])] = c; seen[e[1]] = c + 1

    # 끝단 제거 + 가까운 경계 병합. plateau 와 event 가 겹치면 EVENT 를 남긴다.
    kept = []
    for e in events:
        t = e[0]
        if t < min_seg or t > T - min_seg:
            continue
        if kept and (t - kept[-1][0]) < min_seg:
            if e[2] == "transition" and kept[-1][2] == "plateau":
                kept[-1] = e
            continue
        kept.append(e)
    bounds = [int(e[0]) for e in kept]

    z = np.zeros(T, dtype=int); labels, subgoal_per_phase = [], []
    edges = [0] + bounds + [T]
    for s in range(len(edges) - 1):
        a, b = edges[s], edges[s + 1]
        z[a:b] = s
        closing = [e for e in events if a < e[0] <= b]
        plats = [e for e in closing if e[2] == "plateau"]
        if plats:
            e = plats[-1]
            labels.append(f"{e[1]}:reached")
            # drive-to-min(거리) -> 값을 키우면 degrade (+1)
            sign = +1.0 if _is_decreasing(F[:, e[4]]) else -1.0
            subgoal_per_phase.append((e[1], sign))
        elif closing:
            e = closing[-1]
            engage = (ev_order.get((e[1], e[0]), 0) % 2 == 0)
            labels.append(f"{e[1]}:{'close' if engage else 'open'}")
            # degrade = 전이를 되돌림 (극성 무관)
            sgn = -float(np.sign(e[3]))
            subgoal_per_phase.append((e[1], sgn if sgn != 0 else +1.0))
        else:
            # 어떤 사건도 안 닫은 구간 = tail(retreat). 자기 데이터에서 subgoal 을 읽는다.
            labels.append("final" if s == len(edges) - 2 else f"seg{s}")
            subgoal_per_phase.append(_tail_subgoal(F, a, b, names,
                                                   subgoal_per_phase))
    return dict(z=z, bounds=bounds, labels=labels,
                subgoal_per_phase=subgoal_per_phase, events=events,
                profile_gated=profile_gated)


def _role_in_window(seg, gscale, hold_frac, trend_min):
    """한 phase 창 안에서 feature 하나의 역할: change / hold / free."""
    if len(seg) < 3:
        return "free", 0.0
    rel = float(seg.std()) / max(gscale, 1e-9)
    tv = float(np.abs(np.diff(seg)).sum())
    net = float(seg[-1] - seg[0])
    trend = abs(net) / (tv + 1e-12)          # 1=완전 단조, 0=잡음
    if rel < hold_frac:
        return "hold", 0.0                   # ~일정
    if trend > trend_min:
        return "change", -float(np.sign(net))  # 데모가 몰아감 -> degrade=반대
    return "free", 0.0                       # 진동 -> quality 성격


def phase_subgoal_set(F, z, bounds, main_per_phase, names=None,
                      hold_frac=0.05, trend_min=0.30, pad=2):
    """phase 마다 모든 feature 역할을 데모에서 읽어 subgoal 을 만든다.

        change : 데모가 몰아감 -> degrade=반대(부호)
        hold   : 데모가 유지 -> degrade=흔들기
        free   : 진동 -> subgoal 아님(quality 가 여기로)

    near-binary 그리퍼 phase 에 degrade 대상을 준다: grasp 는 contact 를 change,
    grasp_align / eef_object_dist 를 hold 하므로 그 연속 feature 로 degrade(미정렬/
    접촉 상실, Sec 8.1) 가능.

    pad: phase 의 사건은 닫는 경계 프레임에서 일어난다(그 프레임은 이미 다음 phase 소속).
    창을 경계에서 끊으면 subgoal 이 비므로 bounds+pad 까지 본다.

    quality 는 스키마로 제외(임계값 아님): 큰 저크는 비용이지 phase 실패 아님.
    """
    F = np.asarray(F, float); T = len(F)
    names = list(names) if names is not None else list(fs.NAMES)
    gscale = {n: float(F[:, i].max() - F[:, i].min())
              for i, n in enumerate(names)}
    edges = [0] + list(bounds) + [T]
    out = {}
    for k in range(len(edges) - 1):
        a = edges[k]
        b = min(edges[k + 1] + pad, T)
        change, hold, free = [], [], []
        for i, n in enumerate(names):
            sp = fs.SPEC.get(n)
            if sp is None or sp.kind == fs.QUALITY:
                free.append(n); continue
            role, sgn = _role_in_window(F[a:b, i], gscale[n], hold_frac, trend_min)
            if role == "change":
                change.append((n, sgn))
            elif role == "hold":
                hold.append(n)
            else:
                free.append(n)
        main = main_per_phase[min(k, len(main_per_phase) - 1)]
        # phase 를 닫은 feature 는 main. 창 통계가 못 잡아도(1프레임 사건이 평평해 보임) 넣는다.
        if main[0] not in [c[0] for c in change]:
            change.insert(0, (main[0], float(main[1])))
        hold = [h for h in hold if h != main[0]]
        out[k] = dict(main=(main[0], float(main[1])), change=change, hold=hold,
                      free=free, window=(a, b))
    return out


# ===========================================================================
# SECTION 2 — robosuite replay (실데이터 모드)
# ===========================================================================
_CONTACT_WARNED = False


def read_demo(hdf5_path):
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]; env_info = json.loads(data.attrs["env_info"]); demos = []
        for name in data:
            g = data[name]
            if "states" in g and "actions" in g:
                demos.append((name, g["states"][()], g["actions"][()],
                              g.attrs.get("model_file", None)))
    return env_info, demos


def _robosuite_make(**kwargs):
    try:
        from robosuite import make
    except (ImportError, AttributeError):
        from robosuite.environments.base import make
    return make(**kwargs)


def build_env(env_info):
    return _robosuite_make(
        env_name=env_info["env_name"], robots=env_info["robots"],
        controller_configs=env_info.get("controller_configs"),
        control_freq=env_info.get("control_freq", 20),
        has_renderer=False, has_offscreen_renderer=False,
        ignore_done=True, use_camera_obs=False, reward_shaping=True)


def reset_to_scene(env, model_xml):
    env.reset()
    if model_xml is None:
        return
    xml = model_xml
    try:
        from robosuite.utils.mjcf_utils import postprocess_model_xml
        import inspect
        if len(inspect.signature(postprocess_model_xml).parameters) == 1:
            xml = postprocess_model_xml(model_xml)
    except Exception:
        xml = model_xml
    env.reset_from_xml_string(xml); env.sim.reset(); env.sim.forward()


def _obs(env):
    try:
        return env._get_observations(force_update=True)
    except TypeError:
        return env._get_observations()


def _object_pos(o, object_type):
    """물체의 실제 pose 는 observable '{Name}_pos' 에서. body xpos 는 정적 참조점일 수 있음."""
    if object_type:
        key = f"{object_type.capitalize()}_pos"
        if key in o:
            return np.asarray(o[key], float)
    cand = [k for k in o if k.endswith("_pos")
            and not k.startswith("robot") and "gripper" not in k]
    return np.asarray(o[cand[0]], float) if cand else np.zeros(3)


def _object_quat(o, object_type):
    if object_type:
        key = f"{object_type.capitalize()}_quat"
        if key in o:
            return np.asarray(o[key], float)
    cand = [k for k in o if k.endswith("_quat")
            and not k.startswith("robot") and "gripper" not in k]
    return np.asarray(o[cand[0]], float) if cand else np.array([0.0, 0.0, 0.0, 1.0])


def read_frame(env, object_type):
    o = _obs(env)
    eef = np.asarray(o["robot0_eef_pos"], float)
    grip = float(np.sum(np.abs(np.asarray(o.get("robot0_gripper_qpos", [0.0]),
                                          float))))
    obj = _object_pos(o, object_type)
    eef_q = np.asarray(o.get("robot0_eef_quat", [0.0, 0.0, 0.0, 1.0]), float)
    obj_q = _object_quat(o, object_type)
    return eef, obj, grip, eef_q, obj_q


def resolve_object(env, object_type):
    objs = getattr(env, "objects", None)
    if not objs:
        return None
    if object_type:
        for ob in objs:
            if object_type.lower() in getattr(ob, "name", "").lower():
                return ob
    return objs[0]


def resolve_object_body(env, object_type):
    ob = resolve_object(env, object_type)
    return getattr(ob, "root_body", None) if ob is not None else None


def _gripper_geom_ids(env):
    ids = set()
    try:
        g = env.robots[0].gripper
        names = []
        for attr in ("contact_geoms", "important_geoms"):
            v = getattr(g, attr, None)
            if isinstance(v, dict):
                for vv in v.values():
                    names.extend(vv)
            elif v:
                names.extend(v)
        for n in names:
            try:
                ids.add(env.sim.model.geom_name2id(n))
            except Exception:
                pass
    except Exception:
        pass
    return ids


def _body_geom_ids(env, body_name):
    ids = set()
    try:
        bid = env.sim.model.body_name2id(body_name)
        for gi in range(env.sim.model.ngeom):
            if env.sim.model.geom_bodyid[gi] == bid:
                ids.add(gi)
    except Exception:
        pass
    return ids


def contact_signal(env, obj_model, obj_body):
    """그리퍼가 물체에 닿으면 1.0. 3중 fallback: _check_grasp -> check_contact ->
    geom-pair 수동 스캔. 이게 없으면 상수가 되어 R^2 가 nan."""
    global _CONTACT_WARNED
    try:
        if obj_model is not None and hasattr(env, "_check_grasp"):
            return 1.0 if env._check_grasp(env.robots[0].gripper, obj_model) else 0.0
    except Exception:
        pass
    try:
        if obj_body is not None and hasattr(env, "check_contact"):
            return 1.0 if env.check_contact(env.robots[0].gripper, obj_body) else 0.0
    except Exception:
        pass
    try:
        gg = _gripper_geom_ids(env)
        og = _body_geom_ids(env, obj_body) if obj_body else set()
        if gg and og:
            d = env.sim.data
            for i in range(int(d.ncon)):
                c = d.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in gg and g2 in og) or (g2 in gg and g1 in og):
                    return 1.0
            return 0.0
    except Exception:
        pass
    if not _CONTACT_WARNED:
        print("[warn] contact 판정 불가 -> 0"); _CONTACT_WARNED = True
    return 0.0


def replay(env, states, object_type):
    """데모 state -> frame. M1 이 필요로 하는 단 하나의 replay."""
    obj_model = resolve_object(env, object_type)
    obj_body = resolve_object_body(env, object_type)
    eef, obj, grip, eq, oq, con = [], [], [], [], [], []
    for st in states:
        env.sim.set_state_from_flattened(st); env.sim.forward()
        e, o, g, q1, q2 = read_frame(env, object_type)
        eef.append(e); obj.append(o); grip.append(g); eq.append(q1); oq.append(q2)
        con.append(contact_signal(env, obj_model, obj_body))
    return (np.array(eef), np.array(obj), np.array(grip),
            np.array(eq), np.array(oq), np.array(con))


# (infer_goal / visualize 는 legacy --demo-root 경로 전용이라 제거됨. artifact 흐름에서
#  goal 은 collect_demo 의 goal_context 를 쓰고, 시각화는 각 아티팩트 소비 단계가 담당.)


# ===========================================================================
# SECTION 3 — selftest (합성; robosuite 불필요)
# ===========================================================================
def run_selftest(out_dir="."):
    print("=== phase_segment SELFTEST (합성; robosuite 불필요) ===")
    T, dt = 360, 0.05
    rng = np.random.default_rng(0)

    # eef-물체 거리: approach 에서 ~0 (t=90), 유지, 끝에 상승(후퇴).
    d1 = np.concatenate([np.linspace(1.0, 0.05, 90), np.full(T - 130, 0.05),
                         np.linspace(0.05, 0.45, 40)])
    d1 += rng.normal(0, 0.01, T)
    # 그리퍼: 실제 Panda 극성(닫힐 때 sum|qpos| RISE). 닫힘 t=100, 열림 t=300.
    grip = np.zeros(T); grip[100:300] = 1.0
    # 물체-목표 거리: t=190..290 감소, 230 부근 sidetrack.
    d2 = np.concatenate([np.full(190, 0.5), np.linspace(0.5, 0.02, 100),
                         np.full(T - 290, 0.02)])
    d2[230:250] += np.linspace(0, 0.15, 20)          # regression = sidetrack
    d2[250:270] += np.linspace(0.15, 0, 20)
    # quality: phase 를 만들면 안 됨.
    jerk = rng.normal(0, 1, T) ** 2 * 50
    height = np.concatenate([np.zeros(110), np.linspace(0, 0.2, 80),
                             np.full(T - 190, 0.2)])
    align = np.concatenate([np.linspace(0.8, 0.1, 90), np.full(T - 130, 0.1),
                            np.linspace(0.1, 0.5, 40)]) + rng.normal(0, 0.01, T)
    contact = np.zeros(T); contact[100:300] = 1.0

    cols = {n: np.zeros(T) for n in fs.NON_DEFERRED}
    cols["eef_object_dist"] = d1
    cols["object_goal_dist"] = d2
    cols["gripper_open"] = grip
    cols["contact"] = contact
    cols["grasp_align"] = align
    cols["eef_jerk"] = jerk
    cols["object_height"] = height
    F = np.stack([cols[n] for n in fs.NON_DEFERRED], axis=1)

    seg = segment_features(F)
    nphase = int(seg["z"].max()) + 1
    sset = phase_subgoal_set(F, seg["z"], seg["bounds"], seg["subgoal_per_phase"])

    print(f"\nphases: {nphase}   bounds: {seg['bounds']}")
    for i, (lab, sg) in enumerate(zip(seg["labels"], seg["subgoal_per_phase"])):
        e = [0] + seg["bounds"] + [T]
        d = sset[i]
        ch = ", ".join(f"{n}({s:+.0f})" for n, s in d["change"])
        print(f"  phase {i}: t[{e[i]:3d}:{e[i+1]:3d}]  {lab:<22} main={sg[0]}({sg[1]:+.0f})")
        print(f"          change=[{ch}]")
        print(f"          hold={d['hold']}")

    ok = True
    labs = seg["labels"]

    # 1) 5단계 (approach / grasp / transport / place / retreat) — image 2
    if nphase != 5:
        ok = False; print(f"[FAIL] 5 phase 기대, got {nphase}")

    # 2) 그리퍼가 grasp(닫힘)/place(열림) 경계를 만들고, ORDER 로 라벨
    grip_labs = [l for l in labs if l.startswith("gripper_open:")]
    if grip_labs[:2] != ["gripper_open:close", "gripper_open:open"]:
        ok = False; print(f"[FAIL] 그리퍼 순서 라벨: {grip_labs}")
    else:
        print("\n  그리퍼가 grasp(close)/place(open) 경계를 만듦 ✓")

    # 3) quality(높이/저크)는 phase 를 만들면 안 됨
    if any("object_height" in l for l in labs):
        ok = False; print("[FAIL] object_height(quality)가 phase 생성")
    if any("eef_jerk" in l for l in labs):
        ok = False; print("[FAIL] eef_jerk(quality)가 phase 생성")

    # 4) sidetrack(230..270) 흡수
    leaked = [b for b in seg["bounds"] if 225 <= b <= 275]
    if leaked:
        ok = False; print(f"[FAIL] sidetrack 이 경계를 흘림 @ {leaked}")
    else:
        print("  sidetrack(230..270) 흡수됨 ✓")

    # 5) retreat tail 은 자기 데이터에서 eef_object_dist(-1)
    tf, tsgn = seg["subgoal_per_phase"][-1]
    if tf != "eef_object_dist" or tsgn != -1.0:
        ok = False; print(f"[FAIL] retreat tail 은 eef_object_dist(-1) 이어야, got {tf}({tsgn:+.0f})")
    else:
        print("  retreat tail = eef_object_dist(-1), 자기 데이터에서 읽음 ✓")
        print("    (retreat 는 degrade 해도 task 실패 불가 = pure-efficiency family)")

    # 6) grasp phase(그리퍼가 main)는 degrade 할 연속 멤버가 있어야
    gk = [k for k, d in sset.items() if d["main"][0] == "gripper_open"]
    for k in gk:
        d = sset[k]
        members = [n for n, _ in d["change"] if n != "gripper_open"] + d["hold"]
        cont = [n for n in members if fs.SPEC[n].kind == fs.PROGRESS]
        if not cont:
            ok = False; print(f"[FAIL] 그리퍼 phase {k}: 연속 degrade 멤버 없음")
        else:
            print(f"  그리퍼 phase {k}: 이진 그리퍼 외 {cont} 로 degrade 가능 ✓")

    # 7) quality 는 subgoal 에 못 들어감
    for k, d in sset.items():
        bad = [n for n, _ in d["change"] if fs.SPEC[n].kind == fs.QUALITY]
        bad += [n for n in d["hold"] if fs.SPEC[n].kind == fs.QUALITY]
        if bad:
            ok = False; print(f"[FAIL] phase {k}: quality 가 subgoal 에: {bad}")

    # 8) main 은 항상 change 안에
    for k, d in sset.items():
        if d["main"][0] not in [c[0] for c in d["change"]]:
            ok = False; print(f"[FAIL] phase {k}: main 이 change 에 없음")

    # 9) 잡음 거부
    noise = np.full(T, 0.5) + rng.normal(0, 0.002, T)
    i_eo = fs.NAMES.index("eef_object_dist")
    tr_real, tr_noise = trend_ratio(F[:, i_eo]), trend_ratio(noise)
    print(f"\n  trend_ratio(실제)={tr_real:.2f}  trend_ratio(잡음)={tr_noise:.3f}")
    if tr_noise >= 0.15:
        ok = False; print("[FAIL] 잡음이 progress 필터 통과")

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}  ({nphase} phases, lift 없음)")
    return ok


# ===========================================================================
# SECTION 5 — artifact 흐름: artifacts/features -> artifacts/segmentation
# ===========================================================================
# blueprint v5 Sec 6.7~6.9 파일-단계 흐름. feature_select.py 가 만든 features_*.npz +
# feature_profiles.json 을 읽어 세그먼트/subgoal 을 아티팩트로 쓴다. 세그먼트 코어
# (segment_features / phase_subgoal_set)는 순수 numpy 라 fcm/degradation 이 rollout 궤적을
# 라이브로 재세그먼트할 때도 그대로 재사용된다(단일 진입점).
SEG_SCHEMA_VERSION = "segmentation-1"


def load_feature_profiles(features_dir):
    """feature_select.py 가 쓴 feature_profiles.json. 없으면 None(=profiles 게이트 미사용)."""
    p = os.path.join(features_dir, "feature_profiles.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _sset_to_json(sset):
    """in-memory subgoal set(dict, 'free' 키)를 JSON 친화 구조로. blueprint 용어에 맞춰
    free -> passive 로 노출(단, npz 의 subgoal_set 은 fcm/degradation 계약대로 'free' 유지)."""
    out = {}
    for k, d in sset.items():
        out[str(k)] = {
            "main": {"feature": d["main"][0], "degrade_sign": float(d["main"][1])},
            "change": [{"feature": n, "degrade_sign": float(s)} for n, s in d["change"]],
            "hold": list(d["hold"]),
            "passive": list(d["free"]),          # blueprint 용어(=free)
            "window": [int(d["window"][0]), int(d["window"][1])],
        }
    return out


def _segmentation_gate(seg, sset, goal_from_context, min_seg, max_phases):
    """blueprint Sec 6.9 완료 조건을 단일 데모에서 확인 가능한 범위로 체크.
    goal_from_context 는 경고성(없으면 데모 최종위치 fallback)이라 PASS 를 막지 않는다."""
    z = seg["z"]; nph = int(z.max()) + 1
    edges = [0] + list(seg["bounds"]) + [len(z)]
    seg_lens = [int(edges[i + 1] - edges[i]) for i in range(len(edges) - 1)]
    checks = {
        "phase_count_ok": bool(2 <= nph <= max_phases),
        "min_phase_length_ok": bool(all(L >= min_seg for L in seg_lens)),
        "subgoal_nonempty": bool(all(len(sset[k]["change"]) + len(sset[k]["hold"]) > 0
                                     for k in sset)),
        "no_hardcoded_phase_names": True,        # 라벨은 feature 로부터 파생
        "goal_from_context": bool(goal_from_context),
    }
    checks["PASS"] = bool(checks["phase_count_ok"] and checks["min_phase_length_ok"]
                          and checks["subgoal_nonempty"])
    return dict(n_phases=nph, seg_lengths=seg_lens, checks=checks)


def run_all_features(features_dir, output_dir, seed=42, min_score=0.15, max_phases=8):
    """artifacts/features -> artifacts/segmentation. 데모별 phase_seg_<tag>.npz(기존
    fcm/degradation 이 읽는 스키마와 동일) + subgoals.json + canonical_phases.json +
    segmentation_metrics.json + segmentation_gate.json 을 쓴다."""
    profiles = load_feature_profiles(features_dir)
    paths = sorted(glob(os.path.join(features_dir, "features_*.npz")))
    if not paths:
        raise SystemExit(f"{features_dir} 아래 features_*.npz 없음. "
                         f"먼저 `python feature_select.py all` 실행.")
    os.makedirs(output_dir, exist_ok=True)

    subgoals_json, labels_json = {}, {}
    metrics = {"schema_version": SEG_SCHEMA_VERSION, "n_demos": len(paths),
               "profiles_used": profiles is not None, "min_score": min_score,
               "per_demo": {}}
    gates = {"schema_version": SEG_SCHEMA_VERSION, "per_demo": {}, "PASS": True,
             "goal_context_all": True}

    for p in paths:
        d = np.load(p, allow_pickle=True)
        F = np.asarray(d["features_raw"], float)
        names = [str(x) for x in d["feat_names"]] if "feat_names" in d else list(fs.NAMES)
        dt = float(d["dt"]) if "dt" in d else 0.05
        goal_ctx = bool(d["goal_from_context"]) if "goal_from_context" in d else False
        tag = os.path.splitext(os.path.basename(p))[0].replace("features_", "")

        seg = segment_features(F, names=names, profiles=profiles, min_score=min_score)
        sset = phase_subgoal_set(F, seg["z"], seg["bounds"],
                                 seg["subgoal_per_phase"], names=names)
        min_seg = max(2, int(0.03 * len(F)))
        gate = _segmentation_gate(seg, sset, goal_ctx, min_seg, max_phases)

        np.savez(os.path.join(output_dir, f"phase_seg_{tag}.npz"),
                 z=seg["z"], bounds=np.array(seg["bounds"]),
                 labels=np.array(seg["labels"], dtype=object),
                 subgoal_per_phase=np.array(seg["subgoal_per_phase"], dtype=object),
                 subgoal_set=np.array(sorted(sset.items()), dtype=object),
                 F=F, feat_names=np.array(names, dtype=object), dt=dt,
                 goal=(d["goal"] if "goal" in d else np.array([])),
                 schema_version=SEG_SCHEMA_VERSION)

        subgoals_json[tag] = _sset_to_json(sset)
        labels_json[tag] = list(seg["labels"])
        metrics["per_demo"][tag] = {
            "n_phases": gate["n_phases"], "bounds": list(map(int, seg["bounds"])),
            "seg_lengths": gate["seg_lengths"],
            "profile_gated": [[n, s] for n, s in seg.get("profile_gated", [])]}
        gates["per_demo"][tag] = gate["checks"]
        gates["PASS"] = gates["PASS"] and gate["checks"]["PASS"]
        gates["goal_context_all"] = gates["goal_context_all"] and goal_ctx

    with open(os.path.join(output_dir, "subgoals.json"), "w") as f:
        json.dump(subgoals_json, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "canonical_phases.json"), "w") as f:
        json.dump({"status": "per_demo_only",
                   "note": "multi-demo canonical alignment 는 다음 패스(Sec 6.3/6.7). "
                           "현재는 데모별 독립 phase sequence 만 기록.",
                   "per_demo_phase_labels": labels_json}, f,
                  indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "segmentation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "segmentation_gate.json"), "w") as f:
        json.dump(gates, f, indent=2, ensure_ascii=False)

    print(f"[Stage 6-7 {'PASS' if gates['PASS'] else 'FAIL'}]")
    print(f"Processed demos: {len(paths)}   profiles_used: {profiles is not None}")
    for tag in sorted(metrics["per_demo"]):
        m = metrics["per_demo"][tag]
        print(f"  {tag}: {m['n_phases']} phases  bounds={m['bounds']}  "
              f"labels={labels_json[tag]}")
        if m["profile_gated"]:
            print(f"     profile-gated(구조 약해 경계 제외): {m['profile_gated']}")
    if not gates["goal_context_all"]:
        print("[warn] 일부 demo 에 goal_context 없음 -> object_goal_dist 가 데모 최종위치 "
              "fallback 사용(Sec 6.6). collect_demo extract 의 goal_context 확인 권장.")
    print(f"Artifacts: {output_dir}/ (phase_seg_*.npz, subgoals.json, "
          f"canonical_phases.json, segmentation_metrics.json, segmentation_gate.json)")
    print(f"Next: python fcm.py all --phases {output_dir}")


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="M1: phase 세그먼트 + phase 별 subgoal set "
                    "(artifacts/features -> artifacts/segmentation).")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    pa = sub.add_parser("all", help="artifacts/features -> artifacts/segmentation")
    pa.add_argument("--features", default=os.path.join("artifacts", "features"))
    pa.add_argument("--output", default=os.path.join("artifacts", "segmentation"))
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--min-score", type=float, default=0.15)
    pa.add_argument("--max-phases", type=int, default=8)
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    if args.cmd == "all":
        run_all_features(args.features, args.output, seed=args.seed,
                         min_score=args.min_score, max_phases=args.max_phases)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()