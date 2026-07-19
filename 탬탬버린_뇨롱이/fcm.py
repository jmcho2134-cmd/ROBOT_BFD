#!/usr/bin/env python
"""
fcm.py — M2: Residual Forward Consequence Model (blueprint v5 Sec 7).
================================================================================

책임(Sec 7.2)은 4가지로 제한한다:
    1) baseline-relative counterfactual 잔차 dataset 수집
    2) Residual FCM 학습
    3) 예측 / screening 평가
    4) model checkpoint 저장
degradation hypothesis, exact simulator validation, lambda_max, family 생성은
degradation.py(M3)가 한다. 여기서는 feature/phase 를 정의하지 않고 fs, ps 에서 읽는다.

잔차(Sec 7.4):
    Y = φ_perturbed(t+h) − φ_baseline(t+h)
같은 anchor 에서 두 branch(demo-action baseline / perturbed)를 실행한 차이다.
δ=0 이면 두 branch 가 같아 Y=0 (zero-perturbation gate, Sec 7.9).

입력(Sec 7.5):  [φ_t, a_t, δ_t, z_t, ρ_t, h]  (z=phase id, ρ=phase progress)
split(Sec 7.6): 같은 rollout 의 여러 horizon 이 train/val 로 흩어지지 않게 rollout 단위
                group split (row-shuffle 금지 -- leakage).
평가(Sec 7.7):  예측(MAE/R²/phase·horizon/zero-δ) + screening(Spearman/Top-K recall vs
                random). FCM 은 평균 예측이 아니라 좋은 후보를 Top-K 에 넣는지로 판단.

여기 있는 scoring/candidate 유틸(score_terms, subgoal_score, feature_scales,
action_subspaces, sample_directions, gripper_reverse_direction, phase_span,
phase_progress, is_monotone)은 degradation.py 가 `import fcm as fc` 로 재사용한다(중복 금지).

    python fcm.py --selftest                       # robosuite 불필요(mock)
    python fcm.py all --demos data/demos --raw artifacts/raw \
        --phases artifacts/segmentation --output artifacts/fcm
"""

import argparse
import json
import os
from glob import glob

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import feature_select as fs
import phase_segment as ps


# ===========================================================================
# SECTION 1 — rollout: baseline-relative counterfactual (시뮬레이터 접촉부)
# ===========================================================================
class DemoRollout:
    """demo state t 에 앵커해서 두 branch(demo-action baseline / perturbed)를 실행하고
    baseline 대비 잔차를 반환한다(Sec 7.3–7.4).

    State-anchored: 각 branch 는 env.sim.set_state_from_flattened 로 데모 state t 로
    복원 후 시작한다 -- 실제 하드웨어로는 불가능한, 시뮬레이터 전용 단계다. 앞쪽 몇
    프레임의 데모 이력을 붙여 accel/jerk 를 데모와 같은 이력으로 계산한다.
    """

    def __init__(self, env, states, actions, object_type, dt, frames, goal, pre=3):
        self.env, self.states, self.actions = env, states, actions
        self.object_type, self.dt, self.pre = object_type, dt, pre
        self.frames, self.goal = frames, goal
        self.obj0_z = float(frames["obj"][0, 2])
        self.adim = actions.shape[1]
        self.obj_model = ps.resolve_object(env, object_type)
        self.obj_body = ps.resolve_object_body(env, object_type)
        self.T = len(states)

    def _branch(self, t, delta, H):
        """anchor state t 에서 clip(a_{t+h}+delta) 를 H 스텝 실행. pre-history 를 앞에
        붙여 F 를 만든다. 반환 (F, i0, n, info). baseline 과 perturbed 는 같은 t·pre 를
        쓰므로 i0/n 이 동일 -> 두 branch 가 정확히 정렬된다."""
        f = self.frames
        lo = max(0, t - self.pre); i0 = t - lo
        eef = [f["eef"][j] for j in range(lo, t + 1)]
        obj = [f["obj"][j] for j in range(lo, t + 1)]
        grip = [f["grip"][j] for j in range(lo, t + 1)]
        eq = [f["eef_quat"][j] for j in range(lo, t + 1)]
        oq = [f["obj_quat"][j] for j in range(lo, t + 1)]
        con = [f["contact"][j] for j in range(lo, t + 1)]
        acts = [self.actions[j] for j in range(lo, t)]

        self.env.sim.set_state_from_flattened(self.states[t]); self.env.sim.forward()
        clip_amt, ok = 0.0, True
        for h in range(H):
            a_base = self.actions[min(t + h, len(self.actions) - 1)]
            clip_amt = max(clip_amt, fs.clip_fraction(a_base, delta, self.adim))
            a = fs.perturb_action(a_base, delta, self.adim, proper_rotation=True)
            try:
                self.env.step(a)
            except Exception as ex:
                ok = False; print(f"[warn] step failed at t={t}: {ex}"); break
            e, o, g, q1, q2 = ps.read_frame(self.env, self.object_type)
            eef.append(e); obj.append(o); grip.append(g); eq.append(q1); oq.append(q2)
            con.append(ps.contact_signal(self.env, self.obj_model, self.obj_body))
            acts.append(a)

        n = len(eef)
        if not ok or n <= i0 + 1:
            return None, i0, n, dict(clip=clip_amt, ok=False)
        F = fs.compute_trajectory(
            np.array(eef), np.array(obj), np.array(grip), self.dt,
            np.array(eq), np.array(oq), self.goal, self.obj0_z,
            actions=np.array(acts), contact=np.array(con))
        return F, i0, n, dict(clip=clip_amt, ok=True, contact_end=float(con[-1]))

    def baseline_rollout(self, t, H):
        """demo action branch(δ=0)의 φ_base(t+h), h=0..H. anchor 당 1번 계산해 그 anchor 의
        여러 perturbation 에 재사용(브랜치 비용 절반). 반환 (rows[H+1, N], info)."""
        F, i0, n, info = self._branch(t, np.zeros(self.adim), H)
        if not info["ok"]:
            return None, info
        rows = np.array([F[min(i0 + h, n - 1)] for h in range(0, H + 1)])
        return rows, info

    def counterfactual_rollout(self, t, delta, H, base_rows=None):
        """Sec 7.4 잔차: Y[h] = φ_perturbed(t+h) − φ_baseline(t+h). base_rows 를 주면
        baseline branch 재계산을 건너뛴다. δ=0 → 두 branch 동일 → Y≈0."""
        if base_rows is None:
            base_rows, binfo = self.baseline_rollout(t, H)
            if base_rows is None:
                return np.zeros((H, fs.N_FEATURES)), dict(ok=False,
                                                          clip=binfo.get("clip", 0.0))
        F, i0, n, info = self._branch(t, np.asarray(delta, float), H)
        if not info["ok"]:
            return np.zeros((H, fs.N_FEATURES)), dict(ok=False,
                                                      clip=info.get("clip", 0.0))
        out = np.zeros((H, fs.N_FEATURES))
        for h in range(1, H + 1):
            out[h - 1] = F[min(i0 + h, n - 1)] - base_rows[h]
        return out, dict(clip=info["clip"], ok=True, contact_end=info["contact_end"])


# ===========================================================================
# SECTION 2 — degradation scoring (가중치는 측정값; 손으로 고르지 않음)
# ===========================================================================
def feature_scales(dPhi):
    """수집 데이터에서 feature 별 ΔΦ 표준편차. 거리(m)/각(rad)/그리퍼 같은 비교 불가능한
    양들을 누가 가중치를 고르지 않고 비교 가능하게 만든다."""
    return np.maximum(np.asarray(dPhi, float).std(axis=0), 1e-8)


def score_terms(sset, mode="set", sub_weight=1.0):
    """phase 의 degradation objective 를 [(feature_index, coef, kind), ...] 로.
    kind="signed"(데모가 몰아간 것 반대로) 또는 "abs"(유지한 것 흔들기).
    mode="set": main + 나머지 phase 패턴(change/hold). mode="main": main 만(ablation).
    quality 는 스키마상 제외(큰 저크는 비용이지 phase 실패 아님)."""
    main_name, main_sign = sset["main"]
    terms = [(fs.index_of(main_name), float(main_sign), "signed")]
    if mode == "main":
        return terms
    for nm, sgn in sset["change"]:
        if nm == main_name or nm not in fs.NAMES:
            continue
        terms.append((fs.index_of(nm), float(sub_weight) * float(sgn), "signed"))
    for nm in sset["hold"]:
        if nm in fs.NAMES:
            terms.append((fs.index_of(nm), float(sub_weight), "abs"))
    return terms


def subgoal_score(dPhi, terms, scales):
    """높을수록 phase subgoal 이 더 손상됨. (N,) 또는 (n,N) 수용."""
    d = np.atleast_2d(np.asarray(dPhi, float))
    s = np.zeros(len(d))
    for idx, coef, tk in terms:
        v = d[:, idx] / scales[idx]
        s += coef * (np.abs(v) if tk == "abs" else v)
    return s if len(s) > 1 else float(s[0])


def _spearman(a, b):
    """rank 상관. FCM 예측 순서가 시뮬레이터 실제 순서와 맞았는지 볼 때 사용."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(-a)); rb = np.argsort(np.argsort(-b))
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def r2_per_feature(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        ss = float(((y[:, j] - y[:, j].mean()) ** 2).sum())
        if ss > 1e-12:
            out[j] = 1.0 - float(((y[:, j] - yhat[:, j]) ** 2).sum()) / ss
    return out


# ===========================================================================
# SECTION 3 — the model (multi-output; screening 용)
# ===========================================================================
class FCM:
    """F([φ,a,δ,z,ρ], h) -> ΔΦ^(H). 후보를 RANK 하는 게 목적(정확한 값이 아님) --
    survivor 는 시뮬레이터가 다시 잰다. Z(phase id)/R(phase progress)를 주면 입력에
    포함(Sec 7.5); fit/predict 는 같은 조합을 써야 한다."""

    def __init__(self, hidden=(64, 64), max_iter=800, seed=0):
        self.xs, self.ys = StandardScaler(), StandardScaler()
        self.net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter,
                                random_state=seed, early_stopping=True,
                                n_iter_no_change=20, validation_fraction=0.1)
        self.phase_cond = False

    @staticmethod
    def _col(v, n):
        v = np.atleast_1d(np.asarray(v, float)).reshape(-1, 1)
        return np.repeat(v, n, axis=0) if (len(v) == 1 and n > 1) else v

    @staticmethod
    def _x(C, A, D, h, Z=None, R=None):
        C, A, D = np.atleast_2d(C), np.atleast_2d(A), np.atleast_2d(D)
        n = len(C)
        parts = [C, A, D]
        if Z is not None:
            parts += [FCM._col(Z, n), FCM._col(R, n)]
        parts.append(FCM._col(h, n))
        return np.hstack(parts)

    def fit(self, C, A, D, h, Y, Z=None, R=None):
        self.phase_cond = Z is not None
        X = self._x(C, A, D, h, Z, R)
        self.net.fit(self.xs.fit_transform(X), self.ys.fit_transform(Y))
        return self

    def predict(self, C, A, D, h, Z=None, R=None):
        X = self._x(C, A, D, h, Z, R)
        return self.ys.inverse_transform(self.net.predict(self.xs.transform(X)))


# ===========================================================================
# SECTION 4 — 후보/phase 유틸 (degradation.py 가 fc.* 로 재사용)
# ===========================================================================
def action_subspaces(adim):
    """컨트롤러 자체 분해. OSC_POSE=[dx dy dz | drx dry drz | grip], OSC_POSITION=
    [dx dy dz | grip]. subspace 별로 나눠 탐색하면 위치-미정렬/회전-미정렬이 각자
    자기 사다리를 갖는 별개 family 가 된다(Sec 8.1)."""
    if adim >= 7:
        return [("position", list(range(0, 3))),
                ("rotation", list(range(3, adim - 1))),
                ("gripper", [adim - 1])]
    return [("position", list(range(0, max(adim - 1, 1)))), ("gripper", [adim - 1])]


def sample_directions(M, adim, rng, dims=None):
    """주어진 subspace 단위구 위 균등 샘플(dims=None 이면 전체)."""
    dims = list(range(adim)) if dims is None else list(dims)
    v = rng.normal(0, 1, (M, len(dims)))
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    d = np.zeros((M, adim)); d[:, dims] = v
    return d


def gripper_reverse_direction(actions, a, b, adim):
    """그리퍼 subspace 는 1-D 라 탐색할 게 없다: 두 단위 방향 ±1 중 데모가 명령한 것의
    반대가 degrade 방향."""
    seg = [np.asarray(actions[t], float)[-1] for t in range(a, min(b, len(actions)))]
    g = float(np.sign(np.mean(seg))) if seg else 1.0
    d = np.zeros(adim); d[-1] = -g if g != 0 else 1.0
    return d


def phase_span(z, phase, T):
    """[start, end) of a phase in demo time."""
    idx = np.where(np.asarray(z) == phase)[0]
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1]) + 1


def phase_progress(z, t):
    """ρ_t = (t − phase_start)/(phase_end − phase_start) ∈ [0,1]. blueprint Sec 8.4."""
    z = np.asarray(z); zt = int(z[min(t, len(z) - 1)])
    idx = np.where(z == zt)[0]
    if len(idx) < 2:
        return 0.0
    a, b = int(idx[0]), int(idx[-1])
    return float(np.clip((t - a) / max(b - a, 1), 0.0, 1.0))


def is_monotone(vals, tol_frac=0.07):
    """측정 잡음까지 단조: 곡선 자기 span 의 tol_frac 보다 작은 dip 은 위반 아님."""
    v = np.asarray(vals, float)
    if len(v) < 2:
        return True
    tol = float(tol_frac) * max(float(v.max() - v.min()), 1e-9)
    return bool(np.all(np.diff(v) >= -tol))


# ===========================================================================
# SECTION 5 — dataset(baseline-relative + phase-conditioned) + split + 평가
# ===========================================================================
def collect_dataset(roller, z, C_demo, args, rng, demo_id=0, log=print):
    """anchor 당 baseline branch 를 1번 실행하고 그 위에서 K 개의 perturbation 을
    counterfactual_rollout 로 잰다. rollout_id 는 group split 용. 반환
    dict(C,A,D,Z,R,rollout_id,demo_id,h,Y)."""
    T, adim, H = roller.T, roller.adim, args.horizon
    cols = {k: [] for k in ("C", "A", "D", "Z", "R", "rollout_id", "demo_id", "h", "Y")}
    ts = list(range(0, T - H - 1, max(1, args.subsample)))
    rid = 0
    for i, t in enumerate(ts):
        base_rows, _ = roller.baseline_rollout(t, H)
        if base_rows is None:
            continue
        zt = float(z[min(t, len(z) - 1)]); rho = phase_progress(z, t)
        for k in range(args.perturb_k):
            delta = (np.zeros(adim) if k == 0
                     else rng.normal(0.0, args.perturb_scale, adim))
            dPhi, info = roller.counterfactual_rollout(t, delta, H, base_rows=base_rows)
            if not info["ok"]:
                continue
            for h in range(1, H + 1):
                cols["C"].append(C_demo[t]); cols["A"].append(roller.actions[t])
                cols["D"].append(delta); cols["Z"].append(zt); cols["R"].append(rho)
                cols["rollout_id"].append(rid); cols["demo_id"].append(demo_id)
                cols["h"].append(float(h)); cols["Y"].append(dPhi[h - 1])
            rid += 1
        if i % 20 == 0:
            log(f"    collected {len(cols['Y'])} samples ({i}/{len(ts)} anchors)")
    return {k: np.array(v) for k, v in cols.items()}


def group_split(groups, frac=0.8, seed=0):
    """group(rollout_id 또는 demo_id) 단위 분할. 같은 rollout 의 여러 horizon sample 이
    train/val 에 동시에 들어가지 않게(Sec 7.6 leakage 방지). 반환 (tr_idx, te_idx)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups); rng.shuffle(uniq)
    if len(uniq) <= 1:                        # 단일 그룹: train 에(빈 train 으로 fit crash 방지)
        n_tr = len(uniq)
    else:                                     # 최소 train 1 / test 1 보장
        n_tr = min(max(int(frac * len(uniq)), 1), len(uniq) - 1)
    tr_g = set(uniq[:n_tr].tolist())
    tr = np.array([i for i, g in enumerate(groups) if g in tr_g], dtype=int)
    te = np.array([i for i, g in enumerate(groups) if g not in tr_g], dtype=int)
    return tr, te


def prediction_metrics(ds, tr, te, model):
    """held-out 예측 지표(Sec 7.7): feature-wise MAE/R², phase-wise MAE, horizon-wise
    MAE, subgoal 채널 평균, zero-δ residual. robosuite 불필요(dataset 만)."""
    C, A, D, Z, R, h, Y = (ds["C"], ds["A"], ds["D"], ds["Z"], ds["R"], ds["h"], ds["Y"])
    zr = dict(Z=Z[te], R=R[te]) if getattr(model, "phase_cond", True) else {}
    pred = model.predict(C[te], A[te], D[te], h[te], **zr)
    yt = Y[te]
    mae = np.abs(yt - pred).mean(axis=0)
    r2 = r2_per_feature(yt, pred)
    key = [fs.index_of(n) for n in fs.SUBGOAL_ELIGIBLE]
    zmask = (np.abs(D[te]).sum(axis=1) < 1e-9)
    # zero-δ 진단은 screening 이 쓰는 subgoal 채널로(품질 채널은 스케일이 커서 평균 지배).
    zero_res = (float(np.abs(pred[zmask][:, key]).mean())
                if zmask.any() else float("nan"))

    def _grp_mae(gvals):
        out = {}
        for gv in np.unique(gvals):
            m = gvals == gv
            if m.any():
                out[str(float(gv))] = float(np.abs(yt[m][:, key] - pred[m][:, key]).mean())
        return out

    return dict(
        feature_mae={fs.NAMES[i]: float(mae[i]) for i in range(fs.N_FEATURES)},
        feature_r2={fs.NAMES[i]: (None if np.isnan(r2[i]) else float(r2[i]))
                    for i in range(fs.N_FEATURES)},
        subgoal_r2=float(np.nanmean([r2[i] for i in key])),
        subgoal_mae=float(np.mean([mae[i] for i in key])),
        zero_delta_residual=zero_res,
        phase_mae=_grp_mae(Z[te]), horizon_mae=_grp_mae(h[te]),
        n_train=int(len(tr)), n_test=int(len(te)))


def screening_metrics(model, roller, z, C_demo, args, rng, terms, scales,
                      phase=None, topk=5):
    """screening 지표(Sec 7.7): 후보 M개를 FCM 예측으로 랭킹 vs 시뮬레이터 실제 랭킹.
    Spearman, Top-K recall, best-candidate inclusion, random 대비 향상. 시뮬레이터
    ground-truth 가 필요하므로 roller(robosuite/mock)가 있어야 한다."""
    T, adim, H = roller.T, roller.adim, args.horizon
    if phase is None:
        phase = int(z[len(z) // 2])
    span = phase_span(z, phase, T) or (0, T)
    t = (span[0] + span[1]) // 2
    base_rows, _ = roller.baseline_rollout(t, H)
    if base_rows is None:
        return dict(available=False)
    dirs = sample_directions(args.candidates, adim, rng)
    zt = float(z[min(t, len(z) - 1)]); rho = phase_progress(z, t)
    c0 = C_demo[t].reshape(1, -1); a0 = roller.actions[t].reshape(1, -1)
    pred_s, true_s = [], []
    for d in dirs:
        pr = model.predict(c0, a0, (args.lam_screen * d).reshape(1, -1),
                           float(H), zt, rho)
        pred_s.append(float(subgoal_score(pr[0], terms, scales)))
        dPhi, info = roller.counterfactual_rollout(t, args.lam_screen * d, H,
                                                   base_rows=base_rows)
        true_s.append(float(subgoal_score(dPhi[-1], terms, scales))
                      if info["ok"] else float("-inf"))
    pred_s, true_s = np.array(pred_s), np.array(true_s)
    op, ot = np.argsort(-pred_s), np.argsort(-true_s)
    top_p, top_t = set(op[:topk].tolist()), set(ot[:topk].tolist())
    recall = len(top_p & top_t) / max(len(top_t), 1)
    rand_recall = topk / len(dirs)
    return dict(available=True, phase=int(phase), n_cand=int(len(dirs)),
                spearman=_spearman(pred_s, true_s), topk=int(topk),
                topk_recall=float(recall),
                best_candidate_included=int(ot[0] in top_p),
                random_recall=float(rand_recall),
                lift_over_random=float(recall - rand_recall))


# ===========================================================================
# SECTION 6 — artifact I/O: collect / train / evaluate  (Sec 7.8–7.9)
# ===========================================================================
FCM_SCHEMA_VERSION = "fcm-dataset-1"


def _nan_to_none(o):
    """중첩 dict/list 안의 NaN/Inf float 을 None 으로(표준 JSON 호환). degradation.py 도
    fc._nan_to_none 로 재사용."""
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def save_dataset(output_dir, ds, horizon, score_mode="set"):
    os.makedirs(output_dir, exist_ok=True)
    scales = feature_scales(ds["Y"])
    np.savez(os.path.join(output_dir, "fcm_dataset.npz"),
             C=ds["C"], A=ds["A"], D=ds["D"], Z=ds["Z"], R=ds["R"],
             rollout_id=ds["rollout_id"], demo_id=ds["demo_id"], h=ds["h"], Y=ds["Y"],
             scales=scales, horizon=int(horizon), score_mode=score_mode,
             feat_names=np.array(fs.NAMES, dtype=object),
             schema_version=FCM_SCHEMA_VERSION)
    return scales


def load_dataset(dataset_path):
    d = np.load(dataset_path, allow_pickle=True)
    ds = {k: d[k] for k in ("C", "A", "D", "Z", "R", "rollout_id", "demo_id", "h", "Y")}
    meta = dict(horizon=int(d["horizon"]) if "horizon" in d else 8,
                scales=d["scales"] if "scales" in d else feature_scales(ds["Y"]),
                score_mode=str(d["score_mode"]) if "score_mode" in d else "set")
    return ds, meta


def _load_raw_frames(raw_path):
    """artifacts/raw/raw_<tag>.npz -> DemoRollout frames dict + goal/dt/actions/states.
    degradation.py 도 fc._load_raw_frames 로 재사용."""
    raw = fs.read_raw_npz(raw_path)
    T = len(raw["eef"])
    frames = dict(eef=raw["eef"], obj=raw["obj"], grip=raw["grip"],
                  eef_quat=(raw["eef_q"] if raw["eef_q"] is not None
                            else np.tile([0, 0, 0, 1.0], (T, 1))),
                  obj_quat=(raw["obj_q"] if raw["obj_q"] is not None
                            else np.tile([0, 0, 0, 1.0], (T, 1))),
                  contact=(raw["contact"] if raw["contact"] is not None
                           else np.zeros(T)))
    return frames, raw


def run_train(dataset_path, output_dir, hidden=(64, 64), frac=0.8, seed=0):
    """fcm_dataset.npz -> group split -> FCM 학습 -> fcm_model.joblib(+split). robosuite 불필요.

    커스텀 FCM 클래스 인스턴스를 pickle 하지 않고 sklearn 부품(net/xs/ys)만 저장한다.
    `python fcm.py all` 은 fcm 을 __main__ 으로 실행하므로 FCM 인스턴스를 그대로 저장하면
    __main__.FCM 으로 pickle 되어 degradation.py(다른 __main__)에서 로드가 실패한다. sklearn
    객체는 모듈 경로가 안정적이라 어디서 로드해도 안전하다."""
    ds, _ = load_dataset(dataset_path)
    tr, te = group_split(ds["rollout_id"], frac=frac, seed=seed)
    model = FCM(hidden=hidden, seed=seed).fit(ds["C"][tr], ds["A"][tr], ds["D"][tr],
                                              ds["h"][tr], ds["Y"][tr],
                                              Z=ds["Z"][tr], R=ds["R"][tr])
    os.makedirs(output_dir, exist_ok=True)
    import joblib
    path = os.path.join(output_dir, "fcm_model.joblib")
    joblib.dump(dict(net=model.net, xs=model.xs, ys=model.ys,
                     phase_cond=model.phase_cond, tr=tr, te=te), path)
    print(f"[train] {len(tr)} train / {len(te)} test (group split by rollout_id) -> {path}")
    return model, tr, te


def load_model(path):
    """fcm_model.joblib -> (FCM, tr, te). sklearn 부품에서 FCM 을 재구성한다(커스텀 클래스를
    pickle 하지 않으므로 fcm.py / degradation.py 어느 __main__ 에서든 안전). fcm.py 와
    degradation.py(fc.load_model)가 공용으로 쓴다."""
    import joblib
    blob = joblib.load(path)
    m = FCM.__new__(FCM)                            # __init__ 우회(새 MLP 안 만듦)
    if isinstance(blob, dict) and "net" in blob:    # 현재 포맷(부품)
        m.net, m.xs, m.ys = blob["net"], blob["xs"], blob["ys"]
        m.phase_cond = bool(blob.get("phase_cond", True))
        return m, blob.get("tr"), blob.get("te")
    if isinstance(blob, dict) and "model" in blob:  # 구 포맷(FCM 인스턴스) 하위호환
        return blob["model"], blob.get("tr"), blob.get("te")
    return blob, None, None                          # raw model


def run_evaluate(dataset_path, model_path, output_dir, screening=None):
    """fcm_dataset.npz + fcm_model.joblib -> 예측 지표 -> fcm_metrics.json + fcm_gate.json.
    screening(dict)은 collect 시점(roller 있음)에 계산해 넘겨받는다. robosuite 불필요(예측)."""
    ds, meta = load_dataset(dataset_path)
    model, tr, te = load_model(model_path)
    if tr is None or te is None:               # 부품에 split 이 없으면 재계산
        tr, te = group_split(ds["rollout_id"], frac=0.8, seed=0)
    phase_cond = bool(getattr(model, "phase_cond", True))
    pm = prediction_metrics(ds, tr, te, model)
    sr = pm["subgoal_r2"]; zd = pm["zero_delta_residual"]
    metrics = dict(schema_version=FCM_SCHEMA_VERSION, prediction=pm, screening=screening)
    gate = {
        "subgoal_r2_ok": bool(sr == sr and sr > 0.3),        # NaN -> False
        "zero_delta_ok": bool(zd != zd or zd < 0.05),        # NaN -> 판정 보류(통과)
        "group_split": True,
        "phase_conditioned": phase_cond,
    }
    if screening and screening.get("available"):
        gate["topk_beats_random"] = bool(screening["topk_recall"] > screening["random_recall"])
    gate["PASS"] = bool(gate["subgoal_r2_ok"] and gate.get("topk_beats_random", True))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "fcm_metrics.json"), "w") as f:
        json.dump(_nan_to_none(metrics), f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "fcm_gate.json"), "w") as f:
        json.dump(_nan_to_none(gate), f, indent=2, ensure_ascii=False)
    print(f"[evaluate] subgoal R²={pm['subgoal_r2']:.2f}  "
          f"zero-δ(subgoal)={pm['zero_delta_residual']:.2e}  gate.PASS={gate['PASS']} "
          f"-> {output_dir}/fcm_metrics.json, fcm_gate.json")
    return metrics, gate


def run_collect(demos_dir, raw_dir, phases_dir, output_dir, args, log=print):
    """demo.hdf5(robosuite replay) + artifacts/raw(frames) + artifacts/segmentation(z,
    subgoal_set) -> fcm_dataset.npz (+ screening 지표). robosuite + 실데모 필요.
    ⚠ 실 robosuite 없이 검증 불가 -- baseline/perturbed 두 branch 를 실행한다."""
    seg_paths = sorted(glob(os.path.join(phases_dir, "phase_seg_*.npz")))
    if not seg_paths:
        raise SystemExit(f"{phases_dir} 아래 phase_seg_*.npz 없음. phase_segment all 먼저.")
    demo_paths = sorted(glob(os.path.join(demos_dir, "**", args.pattern), recursive=True))
    demo_by_tag = {os.path.basename(os.path.dirname(p)): p for p in demo_paths}

    cache = {}
    merged = {k: [] for k in ("C", "A", "D", "Z", "R", "rollout_id", "demo_id", "h", "Y")}
    screening = None
    rng = np.random.default_rng(args.seed)
    rid_off, did = 0, 0
    try:
        for sp in seg_paths:
            tag = os.path.basename(sp)[len("phase_seg_"):-len(".npz")]
            raw_path = os.path.join(raw_dir, f"raw_{tag}.npz")
            dp = demo_by_tag.get(tag)                 # 폴더명==tag 매칭(정확한 페어링)
            if not (os.path.exists(raw_path) and dp):
                log(f"[skip] {tag}: raw_{tag}.npz 또는 매칭 demo.hdf5 없음"); continue
            seg = np.load(sp, allow_pickle=True)
            z = seg["z"]; dt = float(seg["dt"])
            frames, raw = _load_raw_frames(raw_path)
            goal = raw["goal"] if raw["goal"] is not None else frames["obj"][-1]
            C_demo = seg["F"]

            env_info, demos = ps.read_demo(dp)
            robots = env_info["robots"]
            sig = (env_info["env_name"], tuple(robots) if isinstance(robots, (list, tuple))
                   else (robots,))
            if sig not in cache:
                log(f"[info] build env {sig[0]} ..."); cache[sig] = ps.build_env(env_info)
            env = cache[sig]; object_type = env_info.get("object_type", None)
            for name, states, actions, model_xml in demos:
                ps.reset_to_scene(env, model_xml)
                roller = DemoRollout(env, states, actions, object_type, dt, frames, goal)
                log(f"[collect] {tag}: baseline+perturbed branches H={args.horizon} ...")
                ds = collect_dataset(roller, z, C_demo, args, rng, demo_id=did, log=log)
                if len(ds["Y"]) == 0:
                    continue
                ds["rollout_id"] = ds["rollout_id"] + rid_off
                rid_off = int(ds["rollout_id"].max()) + 1
                for k in merged:
                    merged[k].append(ds[k])
                if screening is None:            # 첫 데모에서 screening 지표 계산(roller 있음)
                    try:
                        sset_all = {int(k): v for k, v in seg["subgoal_set"]}
                        ph = sorted(sset_all)[len(sset_all) // 2]
                        mdl = FCM().fit(ds["C"], ds["A"], ds["D"], ds["h"], ds["Y"],
                                        Z=ds["Z"], R=ds["R"])
                        terms = score_terms(sset_all[ph], args.score, args.sub_weight)
                        screening = screening_metrics(mdl, roller, z, C_demo, args, rng,
                                                      terms, feature_scales(ds["Y"]), phase=ph)
                    except Exception as ex:
                        log(f"[warn] screening 지표 계산 실패: {ex}")
                did += 1
    finally:
        for env in cache.values():
            try:
                env.close()
            except Exception:
                pass
    if not merged["Y"]:
        raise SystemExit("수집된 sample 이 없음. tag(demo 폴더명) 매칭을 확인하세요.")
    ds = {k: np.concatenate(v) for k, v in merged.items()}
    os.makedirs(output_dir, exist_ok=True)
    save_dataset(output_dir, ds, args.horizon, args.score)
    if screening is not None:
        with open(os.path.join(output_dir, "fcm_screening.json"), "w") as f:
            json.dump(_nan_to_none(screening), f, indent=2, ensure_ascii=False)
    print(f"[Stage 8-9 collect] {len(ds['Y'])} samples, {did} demos "
          f"-> {output_dir}/fcm_dataset.npz")
    return ds, screening


def run_all(demos_dir, raw_dir, phases_dir, output_dir, args):
    ds, screening = run_collect(demos_dir, raw_dir, phases_dir, output_dir, args)
    dp = os.path.join(output_dir, "fcm_dataset.npz")
    run_train(dp, output_dir, seed=args.seed)
    run_evaluate(dp, os.path.join(output_dir, "fcm_model.joblib"), output_dir,
                 screening=screening)
    print(f"Next: python degradation.py all --demos {demos_dir} --raw {raw_dir} "
          f"--phases {phases_dir} --fcm {output_dir} --output artifacts/degradation")


# ===========================================================================
# SECTION 7 — visualization (예측 적합도)
# ===========================================================================
def viz_fit(loss_curve, y, yhat, names, out_png):
    n = len(names); cols = 4; rows = int(np.ceil((n + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    if loss_curve is not None:
        axes[0].plot(loss_curve); axes[0].set_title("training loss")
    r2 = r2_per_feature(y, yhat)
    for j, nm in enumerate(names):
        a = axes[j + 1]
        a.scatter(y[:, j], yhat[:, j], s=6, alpha=0.35)
        lo = float(min(y[:, j].min(), yhat[:, j].min()))
        hi = float(max(y[:, j].max(), yhat[:, j].max()))
        a.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        a.set_title(f"{nm} [{fs.SPEC[nm].kind}]\nR²={r2[j]:.2f}", fontsize=9)
    for a in axes[n + 1:]:
        a.axis("off")
    fig.suptitle("FCM fit: baseline-relative ΔΦ over H steps (screening model)")
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return r2


# ===========================================================================
# SECTION 8 — selftest (mock roller; robosuite 불필요)
# ===========================================================================
class _MockRoller:
    """planted 진실을 가진 toy physics: dim `plant` 만 subgoal 을 손상시키고 lambda 로
    커진다. rollout 은 이미 baseline-relative(δ 의 consequence, baseline 효과 0)."""

    def __init__(self, adim=7, T=200, plant=1, seed=0):
        self.adim, self.T, self.plant = adim, T, plant
        rng = np.random.default_rng(seed)
        self.actions = rng.normal(0, 0.1, (T, adim))
        self.goal = np.array([0.5, 0.0, 0.85])
        self.rng = rng

    def rollout(self, t, delta, H):
        d = np.asarray(delta, float); drive = d[self.plant]
        out = np.zeros((H, fs.N_FEATURES))
        for h in range(1, H + 1):
            r = out[h - 1]
            for nm, gain in (("object_goal_dist", 0.02), ("eef_object_dist", 0.015),
                             ("grasp_align", 0.03), ("contact", -0.01)):
                r[fs.index_of(nm)] = gain * np.tanh(2.0 * drive) * h / 8.0
            r[fs.index_of("eef_jerk")] = 30.0 * float(np.linalg.norm(d)) * h / 8.0
            r[fs.index_of("gripper_open")] = 0.5 * d[-1] * h / 8.0
            r += self.rng.normal(0, 2e-4, fs.N_FEATURES)
        return out, dict(clip=float(fs.clip_fraction(self.actions[t], d, self.adim)),
                         ok=True, contact_end=1.0)

    def baseline_rollout(self, t, H):
        return np.zeros((H + 1, fs.N_FEATURES)), dict(ok=True, clip=0.0)

    def counterfactual_rollout(self, t, delta, H, base_rows=None):
        if np.abs(np.asarray(delta, float)).sum() < 1e-12:   # zero-δ -> 정확히 0
            return np.zeros((H, fs.N_FEATURES)), dict(ok=True, clip=0.0, contact_end=1.0)
        return self.rollout(t, delta, H)


def run_selftest():
    """blueprint Sec 7 검증: baseline-relative 잔차 zero-δ, phase-conditioned 학습,
    group split(누수 없음), 예측/screening 지표, quality-trap. mock; robosuite 불필요."""
    print("=== fcm SELFTEST (mock roller; no robosuite) ===")
    ok = True
    adim, T = 7, 200
    rng = np.random.default_rng(1)
    roller = _MockRoller(adim=adim, T=T, plant=1)

    class A:
        horizon, perturb_k, perturb_scale, subsample = 8, 10, 0.3, 2
        candidates, keep, lam_screen = 40, 3, 1.0
        score, sub_weight = "set", 1.0
    args = A()

    tt = np.linspace(0, 1, T)
    C_demo = rng.normal(0, 0.01, (T, fs.N_FEATURES))
    C_demo[:, fs.index_of("object_goal_dist")] += np.maximum(0.5 - 0.5 * tt, 0.02)
    z = np.zeros(T, dtype=int); z[70:140] = 1; z[140:] = 2

    # 1) zero-δ gate
    base_rows, _ = roller.baseline_rollout(60, args.horizon)
    y0, _ = roller.counterfactual_rollout(60, np.zeros(adim), args.horizon, base_rows=base_rows)
    zmag = float(np.abs(y0).max())
    print(f"[1] zero-δ residual max|Y| = {zmag:.2e}  (baseline-relative -> exact 0)")
    if zmag > 1e-6:
        ok = False; print("[FAIL] zero-δ 잔차가 0 이 아님")

    # 2) phase_progress
    p0, p1 = phase_progress(z, 70), phase_progress(z, 139)
    print(f"[2] phase_progress: t=70 -> {p0:.2f}, t=139 -> {p1:.2f}")
    if not (p0 < 1e-6 and p1 > 0.9):
        ok = False; print("[FAIL] phase_progress 경계값 오류")

    # 3) dataset + group split
    ds = collect_dataset(roller, z, C_demo, args, rng, demo_id=0, log=lambda *a: None)
    tr, te = group_split(ds["rollout_id"], frac=0.8, seed=0)
    leak = set(ds["rollout_id"][tr].tolist()) & set(ds["rollout_id"][te].tolist())
    print(f"[3] dataset {len(ds['Y'])} samples, group leak={len(leak)}")
    if leak or len(ds["Y"]) < 200:
        ok = False; print("[FAIL] group split 누수/표본부족")

    # 4) phase-conditioned 학습 + 예측 지표
    model = FCM(hidden=(64, 64)).fit(ds["C"][tr], ds["A"][tr], ds["D"][tr], ds["h"][tr],
                                     ds["Y"][tr], Z=ds["Z"][tr], R=ds["R"][tr])
    pm = prediction_metrics(ds, tr, te, model)
    print(f"[4] subgoal R²={pm['subgoal_r2']:.2f}  zero-δ pred={pm['zero_delta_residual']:.2e}  "
          f"phase-MAE 키={list(pm['phase_mae'])}")
    if not model.phase_cond:
        ok = False; print("[FAIL] phase_cond 미설정")
    if pm["subgoal_r2"] < 0.4:
        ok = False; print("[FAIL] subgoal R² 너무 낮음(screening 불가)")

    # 5) screening: Top-K recall > random
    scales = feature_scales(ds["Y"])
    sset = dict(main=("object_goal_dist", +1.0), change=[("object_goal_dist", +1.0)],
                hold=["eef_object_dist"], free=[])
    terms = score_terms(sset, "set", 1.0)
    sm = screening_metrics(model, roller, z, C_demo, args, rng, terms, scales, phase=1, topk=5)
    print(f"[5] screening: Top-{sm['topk']} recall={sm['topk_recall']:.2f} "
          f"(random={sm['random_recall']:.2f}, lift={sm['lift_over_random']:+.2f}), "
          f"Spearman={sm['spearman']:.2f}")
    if not sm.get("available") or sm["topk_recall"] <= sm["random_recall"]:
        ok = False; print("[FAIL] Top-K recall 이 random 이하")

    # 6) quality-trap: 품질 feature(jerk)는 subgoal 점수를 못 흔든다
    dj, _ = roller.counterfactual_rollout(100, np.eye(adim)[3], args.horizon)   # 회전=jerk 큼
    dp, _ = roller.counterfactual_rollout(100, np.eye(adim)[1], args.horizon)   # planted subgoal
    s_j = subgoal_score(dj[-1], terms, scales); s_p = subgoal_score(dp[-1], terms, scales)
    print(f"[6] quality-trap: pure-jerk score={s_j:+.2f} vs planted subgoal={s_p:+.2f}")
    if not (s_p > s_j):
        ok = False; print("[FAIL] 품질 feature 가 subgoal 을 이김")
    if any(fs.SPEC[fs.NAMES[i]].kind == fs.QUALITY for i, _, _ in terms):
        ok = False; print("[FAIL] quality feature 가 D_z terms 에")

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ===========================================================================
# main — 단일 흐름: collect / train / evaluate / all / self-test
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="M2: Residual FCM (baseline-relative dataset + train + evaluate).")
    # 튜닝 파라미터(수집/학습). 서브커맨드 앞에 두거나 기본값 사용.
    ap.add_argument("--horizon", type=int, default=8, help="H: sustained-perturbation horizon")
    ap.add_argument("--perturb-k", type=int, default=12, help="anchor 당 perturbation 수(k=0은 δ=0)")
    ap.add_argument("--perturb-scale", type=float, default=0.3)
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=48, help="screening 지표용 후보 수")
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--lam-screen", type=float, default=1.0)
    ap.add_argument("--score", choices=["set", "main"], default="set")
    ap.add_argument("--sub-weight", type=float, default=1.0)
    ap.add_argument("--pattern", default="demo.hdf5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")

    sub = ap.add_subparsers(dest="cmd")
    for nm in ("collect", "all"):
        p = sub.add_parser(nm, help="demo+raw+segmentation -> fcm_dataset"
                                    + (" (+train+evaluate)" if nm == "all" else ""))
        p.add_argument("--demos", default=os.path.join("data", "demos"))
        p.add_argument("--raw", default=os.path.join("artifacts", "raw"))
        p.add_argument("--phases", default=os.path.join("artifacts", "segmentation"))
        p.add_argument("--output", default=os.path.join("artifacts", "fcm"))
    pt = sub.add_parser("train", help="fcm_dataset.npz -> fcm_model.joblib")
    pt.add_argument("--dataset", default=os.path.join("artifacts", "fcm", "fcm_dataset.npz"))
    pt.add_argument("--output", default=os.path.join("artifacts", "fcm"))
    pe = sub.add_parser("evaluate", help="dataset + model -> fcm_metrics/gate.json")
    pe.add_argument("--dataset", default=os.path.join("artifacts", "fcm", "fcm_dataset.npz"))
    pe.add_argument("--model", default=os.path.join("artifacts", "fcm", "fcm_model.joblib"))
    pe.add_argument("--output", default=os.path.join("artifacts", "fcm"))

    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    if args.cmd in ("collect", "all"):
        (run_all if args.cmd == "all" else run_collect)(
            args.demos, args.raw, args.phases, args.output, args)
    elif args.cmd == "train":
        run_train(args.dataset, args.output, seed=args.seed)
    elif args.cmd == "evaluate":
        run_evaluate(args.dataset, args.model, args.output)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
