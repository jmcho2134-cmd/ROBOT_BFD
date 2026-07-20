#!/usr/bin/env python
"""
degradation.py — M3: Structured Degradation (14-feature 파이프라인 재작성).
================================================================================

blueprint v5 Sec 8. subgoals(phase_segment) + Residual FCM(fcm) 로부터 성공을 유지하며
점진적으로 열화되는 Validated Degradation Family 를 만든다.

    Stage 10  hypothesis 생성      change -> reverse_change, hold -> disturb_hold
    Stage 11  candidate 생성 + FCM Top-K screening (탐색 비용 절감)
    Stage 12  exact simulator validation + per-candidate lambda_max + ladder
              + monotonicity/gradedness -> ValidatedDegradationFamily

책임 분리(Sec 7.2): FCM 은 후보를 '선별'만 한다(screening). 최종 degradation 방향/세기는
반드시 실제 simulator rollout(TrajectoryExecutor)로 검증한다. scoring/candidate 유틸은
fcm.py 것을 재사용한다(중복 금지). 이 파일은 feature 나 phase 를 정의하지 않는다.

이전(옛 16-feature/waypoint) 설계 잔재는 제거했다: compute_deferred/assemble/DEFERRED/
waypoint_pos_err/waypoints_from_bounds 는 더 이상 쓰지 않고, 실행 궤적의 feature 는
fs.compute_trajectory(14) 하나로 계산한다.

핵심 게이트: lambda=0 은 반드시 데모를 재현해야 한다(결정론적 replay 확인). 아니면 그
위의 모든 사다리는 신뢰할 수 없다.

    python degradation.py --selftest                       # robosuite 불필요(mock)
    python degradation.py all --demos data/demos --raw artifacts/raw --phases artifacts/segmentation --fcm artifacts/fcm --output artifacts/degradation
"""

import argparse
import json
import os
from glob import glob

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import feature_select as fs
import phase_segment as ps
import fcm as fc


DEG_SCHEMA_VERSION = "degradation-1"
LADDER_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]        # Original/Mild/Medium/Severe/Max
LADDER_NAMES = ["original", "mild", "medium", "severe", "max"]


# ===========================================================================
# SECTION 1 — Stage 10: hypothesis 생성 (Sec 8.3)
# ===========================================================================
def generate_hypotheses(sset, phase):
    """subgoal set -> degradation hypothesis 목록.

    change feature -> reverse_change: 데모가 몰아간 방향을 방해(반대로).
    hold   feature -> disturb_hold : 유지하던 reference 에서 벗어나게.
    passive/free   -> 생성 안 함(직접 degrade 대상 아님, Sec 8.3).
    """
    hyps = []
    for nm, sgn in sset.get("change", []):
        if nm in fs.NAMES:
            hyps.append(dict(phase=int(phase), feature=nm, mode="reverse_change",
                             degrade_sign=float(sgn)))
    for nm in sset.get("hold", []):
        if nm in fs.NAMES:
            hyps.append(dict(phase=int(phase), feature=nm, mode="disturb_hold",
                             degrade_sign=0.0))
    return hyps


# ===========================================================================
# SECTION 2 — Stage 11: candidate 생성 (Sec 8.4)  [fcm 유틸 재사용]
# ===========================================================================
def generate_candidates(actions, z, phase, adim, args, rng):
    """phase 마다 action-subspace 별 후보 방향. fcm.action_subspaces / sample_directions /
    gripper_reverse_direction 재사용. 한 subspace 는 한 종류의 열화(Sec 8.4)."""
    span = fc.phase_span(z, phase, len(z)) or (0, len(z))
    a_, b_ = span
    cands = []
    for sname, dims in fc.action_subspaces(adim):
        if sname == "gripper":
            d = fc.gripper_reverse_direction(actions, a_, b_, adim)
            cands.append(dict(cid=f"p{phase}:{sname}:rev", phase=int(phase),
                              subspace=sname, direction=d))
        else:
            for i, d in enumerate(fc.sample_directions(args.candidates, adim, rng, dims)):
                cands.append(dict(cid=f"p{phase}:{sname}:{i}", phase=int(phase),
                                  subspace=sname, direction=d))
    return cands, span


# ===========================================================================
# SECTION 3 — Stage 11: FCM Top-K screening (Sec 8.5)
# ===========================================================================
def screen_candidates(model, cands, C_demo, actions, z, phase, terms, scales, args):
    """FCM 이 각 후보의 미래 feature residual 을 예측 -> subgoal degradation 점수로 랭킹.
    subspace diversity 를 위해 subspace 별 최소 1개를 먼저 확보한 뒤 점수순으로 채운다."""
    span = fc.phase_span(z, phase, len(z)) or (0, len(z))
    t = (span[0] + span[1]) // 2
    zt = float(z[min(t, len(z) - 1)]); rho = fc.phase_progress(z, t)
    c0 = np.asarray(C_demo[t]).reshape(1, -1); a0 = np.asarray(actions[t]).reshape(1, -1)
    for c in cands:
        d = np.asarray(c["direction"], float)
        pr = model.predict(c0, a0, (args.lam_screen * d).reshape(1, -1),
                           float(args.horizon), zt, rho)
        c["pred_score"] = float(fc.subgoal_score(pr[0], terms, scales))
    ranked = sorted(cands, key=lambda c: -c["pred_score"])
    keep, used_sub = [], set()
    for c in ranked:                                  # subspace 다양성 우선
        if c["subspace"] not in used_sub:
            keep.append(c); used_sub.add(c["subspace"])
        if len(keep) >= args.keep:
            break
    for c in ranked:                                  # 남는 자리 점수순 채움
        if len(keep) >= args.keep:
            break
        if c not in keep:
            keep.append(c)
    return keep, ranked


def optimize_direction_cem(model, C_demo, actions, z, phase, dims, subspace, adim,
                           terms, scales, args, rng, log=print):
    """step (2)의 '학습' 부분. FROZEN FCM 을 surrogate objective 로 써서, subspace 안에서
    subgoal degradation 예측 점수를 '최대화'하는 단위 방향 d 를 CEM 으로 찾는다.
    무작위 후보를 뽑아 필터링(screen_candidates)하던 것을 방향 최적화로 대체.

      maximize   fc.predict_degradation_score(d)         (FCM surrogate)
      s.t.       ||d||=1, d 는 subspace(dims) 안에만 성분

    크기 lam 은 여기서 args.lam_screen 로 고정(방향 랭킹용 작은 probe). 실제 degradation
    크기(lam_max)는 이 방향을 받아 validate_family 가 exact simulator 로 보정한다 --
    FCM 은 큰 magnitude/clipping 근처에서 신뢰도가 낮아 '방향'에만 쓴다(proposal §9.4).
    gripper 는 1-D 라 최적화할 자유도가 없어 부호 뒤집기를 그대로 반환한다."""
    span = fc.phase_span(z, phase, len(z)) or (0, len(z))
    k = len(dims)
    if subspace == "gripper" or k <= 1:
        d = fc.gripper_reverse_direction(actions, span[0], span[1], adim)
        s = fc.predict_degradation_score(model, C_demo, actions, z, phase, d, terms,
                                         scales, args.lam_screen, args.horizon)
        return d, float(s), 1
    # --- CEM: subspace 내 평균 mu / 표준편차 sigma 를 elite 로 반복 재적합 ---
    mu = np.zeros(k)
    sigma = np.full(k, float(args.cem_sigma0))
    best_d, best_s = None, -np.inf
    n_elite = max(2, int(args.cem_elite * args.cem_pop))
    for it in range(args.cem_iters):
        raw = rng.normal(mu, sigma, size=(args.cem_pop, k))          # subspace 내 표본
        raw /= (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12)  # 단위구로 투영
        full = np.zeros((args.cem_pop, adim)); full[:, dims] = raw   # 전체 action 차원으로
        scores = np.array([
            fc.predict_degradation_score(model, C_demo, actions, z, phase, full[i],
                                         terms, scales, args.lam_screen, args.horizon)
            for i in range(args.cem_pop)])
        elite_idx = np.argsort(-scores)[:n_elite]                    # 상위 elite
        mu = raw[elite_idx].mean(axis=0)                             # 분포 재적합
        sigma = raw[elite_idx].std(axis=0) + float(args.cem_eps)
        top = int(elite_idx[0])
        if scores[top] > best_s:
            best_s, best_d = float(scores[top]), full[top].copy()
        log(f"        [p{phase}:{subspace}] CEM it{it} best={scores[top]:.3f} "
            f"sigmā={sigma.mean():.3f}")
    best_d = np.asarray(best_d, float)
    nrm = np.linalg.norm(best_d)
    if nrm > 1e-12:
        best_d = best_d / nrm                                        # 단위 norm 보장
    return best_d, float(best_s), int(args.cem_iters)


# ===========================================================================
# SECTION 4 — perturbed action 시퀀스 + open-loop 실행 (14-feature)
# ===========================================================================
def build_family_actions(actions, z, phase, d_z, lam, adim, family="single"):
    """a'_t = a_t + lam*d_z (family 의 timestep 에서만; 나머지는 데모 action). d_z 는 고정,
    lam 만 사다리에서 변한다(Sec 8.8, Eq 8). 컨트롤러 박스로 클리핑."""
    A = np.array(actions, float).copy()
    d = np.asarray(d_z, float)
    for t in range(len(A)):
        inside = True if family == "all" else (int(z[min(t, len(z) - 1)]) == phase)
        if inside:
            A[t] = fs.perturb_action(A[t], lam * d, adim, proper_rotation=True)
    return A


class TrajectoryExecutor:
    """데모 초기 state 에서 action 시퀀스를 OPEN-LOOP 로 실행(reset 1번 후 step). M2 의
    state-anchored rollout 과 달리 실제 궤적을 만든다 = reward 가 학습할 유일한 궤적
    종류이자 exact simulator validation(Sec 8.6)."""

    def __init__(self, env, states, object_type, dt, goal, obj0_z):
        self.env, self.states, self.object_type = env, states, object_type
        self.dt, self.goal, self.obj0_z = dt, goal, obj0_z
        self.obj_model = ps.resolve_object(env, object_type)
        self.obj_body = ps.resolve_object_body(env, object_type)

    def execute(self, A):
        env = self.env
        env.sim.set_state_from_flattened(self.states[0]); env.sim.forward()
        eef, obj, grip, eq, oq, con = [], [], [], [], [], []

        def snap():
            e, o, g, q1, q2 = ps.read_frame(env, self.object_type)
            eef.append(e); obj.append(o); grip.append(g); eq.append(q1); oq.append(q2)
            con.append(ps.contact_signal(env, self.obj_model, self.obj_body))

        snap(); ok = True
        for t in range(len(A)):
            try:
                env.step(A[t])
            except Exception as ex:
                print(f"[warn] step failed at t={t}: {ex}"); ok = False; break
            snap()
        success = None
        try:
            success = bool(env._check_success())
        except Exception:
            pass
        frames = dict(eef=np.array(eef), obj=np.array(obj), grip=np.array(grip),
                      eef_quat=np.array(eq), obj_quat=np.array(oq),
                      contact=np.array(con))
        return frames, dict(ok=ok, success=success,
                            final_goal_dist=float(np.linalg.norm(
                                frames["obj"][-1] - np.asarray(self.goal, float))))


def executed_features(frames, A, dt, goal, obj0_z):
    """실행 frames -> (T, 14) Phi. waypoint/deferred 없음, compute_trajectory 하나로."""
    return fs.compute_trajectory(frames["eef"], frames["obj"], frames["grip"], dt,
                                 frames["eef_quat"], frames["obj_quat"], goal, obj0_z,
                                 actions=A, contact=frames["contact"])


def reproduces_demo(frames0, demo_eef, tol=0.02):
    """lambda=0 은 데모를 재현해야 한다. 아니면 결정론적 replay 가 아니며 위의 사다리는
    전부 신뢰 불가."""
    e = frames0["eef"]; n = min(len(e), len(demo_eef))
    err = float(np.abs(e[:n] - np.asarray(demo_eef)[:n]).max())
    return err <= tol, err


# ===========================================================================
# SECTION 5 — 한 family 를 한 lambda 에서 실행하고 degradation 점수 측정 (Sec 8.6/8.9)
# ===========================================================================
def run_rung(execu, actions, z, phase, d_z, lam, span, terms, scales, C_demo, dt,
             goal, obj0_z, adim, family="single"):
    """family 를 lam 에서 실행 -> D = subgoal_score(phase-mean(exec) − phase-mean(demo)).
    반환 dict(lam, D, success, clip, ok, frames, A, Phi, final_goal_dist)."""
    A = build_family_actions(actions, z, phase, d_z, lam, adim, family)
    a_, b_ = span
    clip = 0.0
    dvec = lam * np.asarray(d_z, float)
    for t in range(a_, min(b_, len(actions))):
        # build_family_actions 와 동일 조건(실제 perturb 된 step)에서만 clip 집계
        if family == "all" or int(z[min(t, len(z) - 1)]) == phase:
            clip = max(clip, fs.clip_fraction(actions[t], dvec, adim))
    frames, info = execu.execute(A)
    Phi = executed_features(frames, A, dt, goal, obj0_z)
    hi = min(b_, len(Phi))
    if hi <= a_:
        dPhi = np.zeros(fs.N_FEATURES)
    else:
        dPhi = Phi[a_:hi].mean(axis=0) - np.asarray(C_demo)[a_:b_].mean(axis=0)
    return dict(lam=float(lam), D=float(fc.subgoal_score(dPhi, terms, scales)),
                success=info["success"], clip=float(clip), ok=info["ok"],
                frames=frames, A=A, Phi=Phi,
                final_goal_dist=info["final_goal_dist"])


# ===========================================================================
# SECTION 6 — Stage 12: per-candidate lambda_max (Sec 8.7) + ladder (Sec 8.8–8.10)
# ===========================================================================
def search_lambda_max(valid_fn, lam_init=0.1, lam_cap=3.0, tol=0.02, max_iter=20):
    """valid_fn(lam)->bool. bracket(2배씩) 후 이진 탐색으로 성공/feasible 을 유지하는
    최대 lambda. 최소값도 실패면 None. 각 valid_fn 은 실제 simulator 실행 1회."""
    if not valid_fn(lam_init):
        return None
    lo, hi, lam = lam_init, None, lam_init
    while lam < lam_cap:
        lam *= 2.0
        if valid_fn(lam):
            lo = lam
        else:
            hi = lam; break
    if hi is None:
        return lo
    for _ in range(max_iter):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if valid_fn(mid) else (lo, mid)
    return lo


def validate_family(execu, cand, actions, z, phase, span, terms, scales, C_demo, dt,
                    goal, obj0_z, adim, args, demo_eef=None, log=print):
    """한 후보를 exact simulator 로 검증하고 사다리를 만든다.

    1) lambda_max: 성공(=True) + feasible(clip<=max_clip)을 유지하는 최대 lambda
       (bracket+binary). 성공이 None(=env._check_success 판정 불가)이면 valid 아님 -- 확인
       못 한 성공을 성공으로 취급하지 않는다.
    2) ladder: lambda_max x {0,0.25,0.5,0.75,1.0}.
    3) accept: Original..Severe 모두 성공, lambda=0 이 데모 재현(핵심 게이트), D 가 lambda 에
       단조, 레벨 간 gap >= min_gap(gradedness).
    """
    d_z = np.asarray(cand["direction"], float)

    def valid_fn(lam):
        r = run_rung(execu, actions, z, phase, d_z, lam, span, terms, scales,
                     C_demo, dt, goal, obj0_z, adim, args.family)
        # 성공이 확인된(True) 경우만 valid. None(판정 불가)/False 는 valid 아님.
        return bool(r["ok"] and (r["success"] is True) and r["clip"] <= args.max_clip)

    lam_max = search_lambda_max(valid_fn, lam_init=args.lam_init, lam_cap=args.lam_cap)
    if lam_max is None:
        log(f"      [{cand['cid']}] lambda_init={args.lam_init} 에서 성공/feasible 실패 -> 폐기")
        return dict(cid=cand["cid"], phase=int(phase), subspace=cand["subspace"],
                    accepted=False, reason="no_valid_lambda", lam_max=None)

    lambdas = [round(lam_max * f, 5) for f in LADDER_FRACS]
    rungs = [run_rung(execu, actions, z, phase, d_z, lam, span, terms, scales,
                      C_demo, dt, goal, obj0_z, adim, args.family) for lam in lambdas]
    Ds = [r["D"] for r in rungs]
    succ = [r["success"] for r in rungs]
    clips = [r["clip"] for r in rungs]

    # lambda=0 재현(핵심 게이트, docstring). demo_eef 가 없으면 확인 불가 -> None.
    if demo_eef is not None:
        repro, repro_err = reproduces_demo(rungs[0]["frames"], demo_eef)
    else:
        repro, repro_err = None, float("nan")

    # Original..Severe(마지막 Max 제외)에서 성공 유지(성공은 True 여야; None/False 는 불가)
    core = rungs[:-1]
    success_core = all(r["success"] is True for r in core)
    mono = fc.is_monotone([r["D"] for r in core])
    gaps = np.diff([r["D"] for r in core])
    graded = bool(np.all(gaps >= args.min_gap)) and (Ds[-2] - Ds[0] >= args.min_effect)
    accepted = bool(success_core and mono and graded and (repro is not False))

    log(f"      [{cand['cid']}] lam_max={lam_max:.3f}  D={np.round(Ds, 2)}  "
        f"succ={''.join('T' if s else ('F' if s is False else '?') for s in succ)}  "
        f"mono={'O' if mono else 'X'} graded={'O' if graded else 'X'} "
        f"repro={'O' if repro else ('X' if repro is False else '?')} "
        f"-> {'ACCEPT' if accepted else 'reject'}")

    return dict(cid=cand["cid"], phase=int(phase), subspace=cand["subspace"],
                accepted=accepted, lam_max=float(lam_max), lambdas=lambdas,
                D=[float(x) for x in Ds], success=[None if s is None else bool(s) for s in succ],
                clip=[float(c) for c in clips], direction=d_z,
                monotone=bool(mono), graded=bool(graded), success_core=bool(success_core),
                reproduces_demo=(None if repro is None else bool(repro)),
                repro_err=float(repro_err),
                pred_score=float(cand.get("pred_score", 0.0)), rungs=rungs)


# ===========================================================================
# SECTION 7 — preference pairs (현재 범위 밖: 후속 reward 용으로 보존, Sec 8.2)
# ===========================================================================
def make_preference_pairs(family, family_id):
    """lam_i < lam_j => tau_i > tau_j. family 내부에서만. 현재 main 최종출력엔 미포함
    (Stage 13+ reward 단계용). label 은 lambda 에서만 온다."""
    lam = family["lambdas"]
    pairs = []
    for i in range(len(lam)):
        for j in range(i + 1, len(lam)):
            if lam[i] < lam[j]:
                pairs.append(dict(family=family_id, better=i, worse=j,
                                  lam_better=lam[i], lam_worse=lam[j]))
    return pairs


# ===========================================================================
# SECTION 8 — visualization
# ===========================================================================
def viz_families(families, demo_frames, goal, z_demo, out_png):
    acc = [f for f in families if f.get("accepted")]
    if not acc:
        return
    n = len(acc); cols = min(3, n); rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(6 * cols, 5 * rows)); cmap = plt.get_cmap("plasma")
    for i, f in enumerate(acc):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        de = demo_frames["eef"]
        ax.plot(de[:, 0], de[:, 1], de[:, 2], color="black", linewidth=2.5, label="demo")
        lm = f["lam_max"] + 1e-9
        for r in f["rungs"]:
            if r["lam"] == 0:
                continue
            c = cmap(0.15 + 0.75 * r["lam"] / lm); e = r["frames"]["eef"]
            ax.plot(e[:, 0], e[:, 1], e[:, 2], color=c, linewidth=1.5,
                    label=f"λ={r['lam']}")
        ax.scatter(*goal, color="red", marker="*", s=150, label="goal")
        ax.set_title(f"phase {f['phase']} / {f['subspace']}\nD={np.round(f['D'],2)}",
                     fontsize=9)
        ax.legend(fontsize=6, loc="upper left")
    fig.suptitle("Validated Degradation Families (executed open-loop, simulator-validated)")
    fig.tight_layout(); fig.savefig(out_png, dpi=125, bbox_inches="tight"); plt.close(fig)


# ===========================================================================
# SECTION 9 — artifact 흐름: hypothesize / screen / validate
# ===========================================================================
def _load_scales(fcm_dir):
    p = os.path.join(fcm_dir, "fcm_dataset.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        if "scales" in d:
            return np.asarray(d["scales"], float)
    return None


def _load_fcm_model(path):
    """fcm_model.joblib 로드. fc.load_model 은 sklearn 부품에서 FCM 을 재구성하므로 fcm 을
    __main__ 으로 학습·저장했어도(=`python fcm.py all`) 안전하게 로드된다."""
    return fc.load_model(path)[0]


def run_hypothesize(phases_dir, output_dir):
    """artifacts/segmentation -> degradation_hypotheses.json (robosuite 불필요)."""
    seg_paths = sorted(glob(os.path.join(phases_dir, "phase_seg_*.npz")))
    if not seg_paths:
        raise SystemExit(f"{phases_dir} 아래 phase_seg_*.npz 없음. phase_segment all 먼저.")
    os.makedirs(output_dir, exist_ok=True)
    out = {}
    for sp in seg_paths:
        tag = os.path.basename(sp)[len("phase_seg_"):-len(".npz")]
        seg = np.load(sp, allow_pickle=True)
        sset_all = {int(k): v for k, v in seg["subgoal_set"]}
        hyps = []
        for ph in sorted(sset_all):
            hyps += generate_hypotheses(sset_all[ph], ph)
        out[tag] = hyps
    with open(os.path.join(output_dir, "degradation_hypotheses.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    n = sum(len(v) for v in out.values())
    print(f"[hypothesize] {n} hypotheses across {len(out)} demos "
          f"-> {output_dir}/degradation_hypotheses.json")
    return out


def run_all_degradation(demos_dir, raw_dir, phases_dir, fcm_dir, output_dir, args):
    """전체 M3: 각 demo/phase 마다 hypothesis -> 방향 결정 -> exact validation -> lambda_max
    -> ladder -> validated family. exact validation 은 robosuite 필요.

    방향 결정(step 2)은 args.search 로 선택:
      cem(기본)  : FROZEN FCM 을 surrogate 로 subgoal 열화 방향을 CEM 으로 '최적화'
      random     : 기존 무작위 후보 + FCM Top-K 필터(ablation, proposal §13.2)
    크기(lambda_max)는 두 경우 모두 exact simulator 가 validate_family 에서 보정한다."""
    seg_paths = sorted(glob(os.path.join(phases_dir, "phase_seg_*.npz")))
    if not seg_paths:
        raise SystemExit(f"{phases_dir} 아래 phase_seg_*.npz 없음.")
    model_path = os.path.join(fcm_dir, "fcm_model.joblib")
    if not os.path.exists(model_path):
        raise SystemExit(f"{model_path} 없음. fcm.py all/train 먼저.")
    model = _load_fcm_model(model_path)
    scales = _load_scales(fcm_dir)
    demo_paths = sorted(glob(os.path.join(demos_dir, "**", args.pattern), recursive=True))
    demo_by_tag = {os.path.basename(os.path.dirname(p)): p for p in demo_paths}
    os.makedirs(output_dir, exist_ok=True)

    hyps_all, screened_all, lam_all, families_all, metrics = {}, {}, {}, [], {}
    cache = {}
    rng = np.random.default_rng(args.seed)
    try:
        for sp in seg_paths:
            tag = os.path.basename(sp)[len("phase_seg_"):-len(".npz")]
            raw_path = os.path.join(raw_dir, f"raw_{tag}.npz")
            dp = demo_by_tag.get(tag)
            if not (os.path.exists(raw_path) and dp):
                print(f"[skip] {tag}: raw 또는 demo.hdf5 없음"); continue
            seg = np.load(sp, allow_pickle=True)
            z = seg["z"]; dt = float(seg["dt"]); C_demo = seg["F"]
            sset_all = {int(k): v for k, v in seg["subgoal_set"]}
            frames, raw = fc._load_raw_frames(raw_path)
            goal = raw["goal"] if raw["goal"] is not None else frames["obj"][-1]
            obj0_z = float(frames["obj"][0, 2])
            sc = scales if scales is not None else np.ones(fs.N_FEATURES)

            env_info, demos = ps.read_demo(dp)
            robots = env_info["robots"]
            sig = (env_info["env_name"], tuple(robots) if isinstance(robots, (list, tuple))
                   else (robots,))
            if sig not in cache:
                print(f"[info] build env {sig[0]} ..."); cache[sig] = ps.build_env(env_info)
            env = cache[sig]; object_type = env_info.get("object_type", None)

            for name, states, actions, model_xml in demos:
                ps.reset_to_scene(env, model_xml)
                adim = actions.shape[1]
                execu = TrajectoryExecutor(env, states, object_type, dt, goal, obj0_z)
                demo_eef = frames["eef"]
                hyps_all.setdefault(tag, []); screened_all.setdefault(tag, [])
                for ph in sorted(sset_all):
                    sset = sset_all[ph]
                    hyps = generate_hypotheses(sset, ph)
                    hyps_all[tag] += hyps
                    if not hyps:
                        continue
                    terms = fc.score_terms(sset, args.score, args.sub_weight)
                    span = fc.phase_span(z, ph, len(z)) or (0, len(z))
                    if args.search == "cem":
                        # step (2): subspace 마다 FCM surrogate 로 열화 방향을 '최적화'
                        keep = []
                        for sname, dims in fc.action_subspaces(adim):
                            d_opt, s_opt, _ = optimize_direction_cem(
                                model, C_demo, actions, z, ph, dims, sname, adim,
                                terms, sc, args, rng)
                            keep.append(dict(cid=f"p{ph}:{sname}:cem", phase=int(ph),
                                             subspace=sname, direction=d_opt,
                                             pred_score=float(s_opt)))
                        print(f"\n[phase {ph}] CEM-optimized {len(keep)} directions "
                              f"({[c['cid'] for c in keep]})")
                    else:
                        # ablation: 기존 무작위 후보 + FCM Top-K 필터 (proposal §13.2)
                        cands, span = generate_candidates(actions, z, ph, adim, args, rng)
                        keep, _ = screen_candidates(model, cands, C_demo, actions, z,
                                                    ph, terms, sc, args)
                        print(f"\n[phase {ph}] {len(cands)} random cands -> Top-{len(keep)} "
                              f"({[c['cid'] for c in keep]})")
                    screened_all[tag] += [dict(cid=c["cid"], subspace=c["subspace"],
                                               pred_score=c["pred_score"]) for c in keep]
                    for c in keep:
                        fam = validate_family(execu, c, actions, z, ph, span, terms, sc,
                                              C_demo, dt, goal, obj0_z, adim, args,
                                              demo_eef=demo_eef)
                        fam["tag"] = tag
                        lam_all[c["cid"]] = fam.get("lam_max")
                        families_all.append(fam)
    finally:
        for env in cache.values():
            try:
                env.close()
            except Exception:
                pass

    accepted = [f for f in families_all if f.get("accepted")]
    _save_degradation(output_dir, hyps_all, screened_all, lam_all, families_all,
                      accepted, args)
    n_hyp = sum(len(v) for v in hyps_all.values())
    yield_ = len(accepted) / max(len(families_all), 1)
    print(f"\n[Stage 10-12 {'PASS' if accepted else 'FAIL'}]")
    print(f"Validated families: {len(accepted)}/{len(families_all)} "
          f"(yield={yield_:.2f}) from {n_hyp} hypotheses")
    print(f"Artifacts: {output_dir}/ (validated_families.json/.npz, "
          f"degradation_hypotheses.json, screened_candidates.json, "
          f"lambda_search_results.json, degradation_metrics.json, degradation_gate.json)")
    return accepted


def _save_degradation(output_dir, hyps, screened, lam, families, accepted, args):
    with open(os.path.join(output_dir, "degradation_hypotheses.json"), "w") as f:
        json.dump(hyps, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "screened_candidates.json"), "w") as f:
        json.dump(screened, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "lambda_search_results.json"), "w") as f:
        json.dump({k: (None if v is None else float(v)) for k, v in lam.items()},
                  f, indent=2, ensure_ascii=False)
    fam_json = [dict(tag=f.get("tag"), cid=f["cid"], phase=f["phase"],
                     subspace=f["subspace"], accepted=f["accepted"],
                     lam_max=f.get("lam_max"), lambda_values=f.get("lambdas"),
                     actual_degradation_scores=f.get("D"), success_flags=f.get("success"),
                     clip_fractions=f.get("clip"), monotone=f.get("monotone"),
                     graded=f.get("graded"),
                     direction=[float(x) for x in np.asarray(f.get("direction", []))],
                     reproduces_demo=f.get("reproduces_demo"))
                for f in families]
    with open(os.path.join(output_dir, "validated_families.json"), "w") as f:
        json.dump(fc._nan_to_none(fam_json), f, indent=2, ensure_ascii=False)
    # 실제 궤적(accepted 만) npz 저장
    save = {}
    for i, f in enumerate(accepted):
        for j, r in enumerate(f["rungs"]):
            save[f"Phi_{i}_{j}"] = r["Phi"]; save[f"eef_{i}_{j}"] = r["frames"]["eef"]
            save[f"obj_{i}_{j}"] = r["frames"]["obj"]; save[f"A_{i}_{j}"] = r["A"]
    np.savez(os.path.join(output_dir, "validated_families.npz"),
             feat_names=np.array(fs.NAMES, dtype=object),
             families=np.array(fam_json, dtype=object),
             schema_version=DEG_SCHEMA_VERSION, **save)
    yield_ = len(accepted) / max(len(families), 1)
    mono_rate = (sum(f.get("monotone", False) for f in families) / max(len(families), 1))
    metrics = dict(schema_version=DEG_SCHEMA_VERSION, n_hypotheses=sum(len(v) for v in hyps.values()),
                   n_candidates_validated=len(families), n_accepted=len(accepted),
                   family_yield=yield_, monotonicity_rate=mono_rate)
    with open(os.path.join(output_dir, "degradation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    gate = dict(schema_version=DEG_SCHEMA_VERSION,
                any_validated_family=bool(accepted),
                all_accepted_monotone=all(f.get("monotone") for f in accepted),
                all_accepted_graded=all(f.get("graded") for f in accepted),
                all_accepted_success_core=all(f.get("success_core") for f in accepted),
                all_accepted_reproduce_demo=all(f.get("reproduces_demo") is True
                                                for f in accepted))
    # 핵심 게이트(docstring): lambda=0 재현 + Original..Severe 성공 + monotone + graded
    gate["PASS"] = bool(gate["any_validated_family"] and gate["all_accepted_monotone"]
                        and gate["all_accepted_graded"]
                        and gate["all_accepted_success_core"]
                        and gate["all_accepted_reproduce_demo"])
    with open(os.path.join(output_dir, "degradation_gate.json"), "w") as f:
        json.dump(gate, f, indent=2, ensure_ascii=False)
    if accepted:
        try:
            df0 = accepted[0]["rungs"][0]["frames"]
            viz_families(accepted, dict(eef=df0["eef"], obj=df0["obj"]),
                         accepted[0]["rungs"][0]["frames"]["obj"][-1], None,
                         os.path.join(output_dir, "validated_families.png"))
        except Exception as ex:
            print(f"[warn] viz 실패: {ex}")


# ===========================================================================
# SECTION 10 — selftest (mock; robosuite 불필요)
# ===========================================================================
class _MockExecutor:
    """open-loop toy physics: eef 가 action 을 적분, grasped 동안 object 가 eef 를 따라감."""

    def __init__(self, T=200, adim=7, goal=np.array([0.5, 0.0, 0.85])):
        self.T, self.adim, self.goal, self.obj0_z = T, adim, goal, 0.85

    def execute(self, A):
        T = len(A)
        eef = np.zeros((T + 1, 3)); eef[0] = [0.0, 0.0, 1.0]
        obj = np.zeros((T + 1, 3)); obj[0] = [0.25, 0.0, 0.85]
        grip = np.zeros(T + 1); con = np.zeros(T + 1)
        for t in range(T):
            eef[t + 1] = eef[t] + 0.01 * np.asarray(A[t], float)[:3]
            grasped = (T * 0.3 < t < T * 0.85)
            grip[t + 1] = con[t + 1] = 1.0 if grasped else 0.0
            obj[t + 1] = (obj[t] + (eef[t + 1] - eef[t])) if grasped else obj[t]
        q = np.tile([0.0, 0.0, 0.0, 1.0], (T + 1, 1))
        frames = dict(eef=eef, obj=obj, grip=grip, eef_quat=q, obj_quat=q, contact=con)
        fd = float(np.linalg.norm(obj[-1] - self.goal))
        return frames, dict(ok=True, success=bool(fd < 0.12), final_goal_dist=fd)


class _MockScreenModel:
    """screening 이 '돈다'만 확인하는 mock FCM: position -x(=object 를 goal 반대로) 일수록
    object_goal_dist 가 커진다고 예측 -> reverse_change(object_goal_dist,+) 점수 높아짐."""

    def predict(self, C, A, D, h, Z=None, R=None):
        D = np.atleast_2d(np.asarray(D, float))
        out = np.zeros((len(D), fs.N_FEATURES))
        out[:, fs.index_of("object_goal_dist")] = -D[:, 0]      # -x -> 거리 증가
        return out


def run_selftest(out_dir="."):
    print("=== degradation SELFTEST (mock physics; no robosuite) ===")
    ok = True
    T, adim, dt = 200, 7, 0.05

    class A:
        candidates, keep, horizon, lam_screen = 24, 3, 8, 1.0
        score, sub_weight = "set", 1.0
        lam_init, lam_cap, max_clip = 0.1, 3.0, 0.6
        min_gap, min_effect, family, seed = 0.02, 0.05, "single", 0
        search = "cem"
        cem_pop, cem_iters, cem_elite, cem_sigma0, cem_eps = 64, 8, 0.2, 1.0, 1e-3
    args = A()

    # 데모: +x 로 object 를 goal 로 옮김(성공). phase 1 이 transport.
    actions = np.zeros((T, adim)); actions[:, 0] = 0.25
    z = np.zeros(T, dtype=int); z[60:170] = 1; z[170:] = 2
    execu = _MockExecutor(T=T, adim=adim); goal = execu.goal
    demo_frames, _ = execu.execute(actions)
    C_demo = executed_features(demo_frames, actions, dt, goal, execu.obj0_z)
    scales = np.ones(fs.N_FEATURES) * 0.05
    rng = np.random.default_rng(0)

    # subgoal set: transport phase 는 object_goal_dist 를 줄임(change,+1), eef_object_dist hold
    sset = dict(main=("object_goal_dist", +1.0),
                change=[("object_goal_dist", +1.0)], hold=["eef_object_dist"],
                free=["eef_jerk"])
    terms = fc.score_terms(sset, "set", 1.0)

    # 1) hypothesis: change->reverse_change, hold->disturb_hold, passive 없음
    hyps = generate_hypotheses(sset, 1)
    modes = sorted(set(h["mode"] for h in hyps))
    print(f"[1] hypotheses: {[(h['feature'], h['mode']) for h in hyps]}")
    if modes != ["disturb_hold", "reverse_change"]:
        ok = False; print("[FAIL] hypothesis 모드 집합 오류")
    if any(h["feature"] == "eef_jerk" for h in hyps):
        ok = False; print("[FAIL] passive(quality) feature 로 hypothesis 생성됨")

    # 2) candidates: subspace 별 존재
    cands, span = generate_candidates(actions, z, 1, adim, args, rng)
    subs = sorted(set(c["subspace"] for c in cands))
    print(f"[2] candidates: {len(cands)}개, subspaces={subs}")
    if subs != sorted(n for n, _ in fc.action_subspaces(adim)):
        ok = False; print("[FAIL] subspace 누락")

    # 3) screening: Top-K, subspace 다양성
    keep, ranked = screen_candidates(_MockScreenModel(), cands, C_demo, actions, z, 1,
                                     terms, scales, args)
    print(f"[3] Top-{len(keep)}: {[c['cid'] for c in keep]}  "
          f"(pred_scores={[round(c['pred_score'],2) for c in keep]})")
    if len(keep) != args.keep or len(set(c['subspace'] for c in keep)) < 2:
        ok = False; print("[FAIL] Top-K 개수/다양성 오류")

    # 4) validate: -x 방향 후보로 사다리 -> mono/graded/success
    neg_x = dict(cid="test:position:negx", phase=1, subspace="position",
                 direction=np.array([-1.0, 0, 0, 0, 0, 0, 0]), pred_score=1.0)
    fam = validate_family(execu, neg_x, actions, z, 1, span, terms, scales, C_demo, dt,
                          goal, execu.obj0_z, adim, args, demo_eef=demo_frames["eef"],
                          log=lambda *a: None)
    print(f"[4] family: lam_max={fam.get('lam_max')}  D={np.round(fam.get('D', []),2)}  "
          f"succ={fam.get('success')}  mono={fam.get('monotone')} "
          f"graded={fam.get('graded')} -> accepted={fam['accepted']}")
    if fam["lam_max"] is None:
        ok = False; print("[FAIL] lambda_max 탐색 실패")
    else:
        Ds = fam["D"]
        if not (Ds[3] > Ds[0]):
            ok = False; print("[FAIL] degradation 이 lambda 로 증가 안 함")
        if not fam["monotone"]:
            ok = False; print("[FAIL] 사다리가 단조가 아님")
        if not all(s is not False for s in fam["success"][:4]):
            ok = False; print("[FAIL] Original..Severe 중 실패 rung 존재")

    # 5) lambda=0 재현
    rep, err = reproduces_demo(fam["rungs"][0]["frames"], demo_frames["eef"])
    print(f"[5] lambda=0 reproduces demo: {'✓' if rep else '✗'} (err={err:.4f})")
    if not rep:
        ok = False; print("[FAIL] lambda=0 이 데모를 재현 안 함")

    # 6) d_z 고정(사다리 전체에서 동일)
    dirs = [r["A"][z[:T] == 1][0] - actions[z[:T] == 1][0] for r in fam["rungs"][1:]]
    unit = [dd / (np.linalg.norm(dd) + 1e-12) for dd in dirs]
    fixed = all(np.allclose(u, unit[0], atol=1e-6) for u in unit)
    print(f"[6] d_z fixed across ladder (Eq 8): {'✓' if fixed else '✗'}")
    if not fixed:
        ok = False; print("[FAIL] 사다리에서 d_z 방향이 변함")

    # 7) quality(passive) 는 절대 subgoal terms 에 없음
    if any(fs.SPEC[fs.NAMES[i]].kind == fs.QUALITY for i, _, _ in terms):
        ok = False; print("[FAIL] quality feature 가 degradation 점수에 들어감")

    # 8) step (2) CEM: mock 은 -x 일수록 object_goal_dist 증가 -> CEM 이 position 을 -x 로 찾아야
    pos_dims = dict(fc.action_subspaces(adim))["position"]
    d_cem, s_cem, _ = optimize_direction_cem(_MockScreenModel(), C_demo, actions, z, 1,
                                             pos_dims, "position", adim, terms, scales,
                                             args, rng, log=lambda *a: None)
    # 무작위 방향 평균 점수와 비교(최적화가 랜덤보다 나아야)
    rand_scores = []
    for _ in range(64):
        rd = np.zeros(adim); rd[pos_dims] = rng.normal(0, 1, len(pos_dims))
        rd /= (np.linalg.norm(rd) + 1e-12)
        rand_scores.append(fc.predict_degradation_score(
            _MockScreenModel(), C_demo, actions, z, 1, rd, terms, scales, 1.0, 8))
    print(f"[8] CEM dir={np.round(d_cem[pos_dims],2)} score={s_cem:.3f}  "
          f"vs random mean={np.mean(rand_scores):.3f}  (x성분={d_cem[0]:+.2f})")
    if d_cem[0] >= -0.5:
        ok = False; print("[FAIL] CEM 이 열화 방향(-x)을 못 찾음")
    if s_cem <= np.mean(rand_scores):
        ok = False; print("[FAIL] CEM 이 무작위 평균보다 나쁨")

    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="M3: structured degradation families.")
    ap.add_argument("--pattern", default="demo.hdf5")
    # 탐색/검증 파라미터
    ap.add_argument("--candidates", type=int, default=48, help="후보 수(subspace 당)")
    ap.add_argument("--keep", type=int, default=4, help="FCM screening Top-K")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--lam-screen", type=float, default=1.0)
    ap.add_argument("--lam-init", type=float, default=0.05)
    ap.add_argument("--lam-cap", type=float, default=3.0)
    ap.add_argument("--max-clip", type=float, default=0.5)
    ap.add_argument("--min-gap", type=float, default=0.02,
                    help="레벨 간 최소 degradation 차이(gradedness)")
    ap.add_argument("--min-effect", type=float, default=0.05,
                    help="Original->Severe 최소 총 degradation")
    ap.add_argument("--score", choices=["set", "main"], default="set")
    ap.add_argument("--sub-weight", type=float, default=1.0)
    # step (2) 방향 결정 방식: cem=FCM surrogate 최적화(기본), random=기존 무작위+필터(ablation §13.2)
    ap.add_argument("--search", choices=["cem", "random"], default="cem",
                    help="열화 방향을 최적화(cem)할지 무작위 대입(random)할지")
    ap.add_argument("--cem-pop", type=int, default=64, help="CEM 세대 표본 수")
    ap.add_argument("--cem-iters", type=int, default=8, help="CEM 반복 횟수")
    ap.add_argument("--cem-elite", type=float, default=0.2, help="elite 비율")
    ap.add_argument("--cem-sigma0", type=float, default=1.0, help="CEM 초기 표준편차")
    ap.add_argument("--cem-eps", type=float, default=1e-3, help="sigma 하한(조기수렴 방지)")
    ap.add_argument("--family", choices=["single", "all"], default="single")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")

    sub = ap.add_subparsers(dest="cmd")
    ph = sub.add_parser("hypothesize", help="segmentation -> hypotheses.json (robosuite 불필요)")
    ph.add_argument("--phases", default=os.path.join("artifacts", "segmentation"))
    ph.add_argument("--output", default=os.path.join("artifacts", "degradation"))
    pa = sub.add_parser("all", help="hypothesis->screen->validate->families (robosuite 필요)")
    pa.add_argument("--demos", default=os.path.join("data", "demos"))
    pa.add_argument("--raw", default=os.path.join("artifacts", "raw"))
    pa.add_argument("--phases", default=os.path.join("artifacts", "segmentation"))
    pa.add_argument("--fcm", default=os.path.join("artifacts", "fcm"))
    pa.add_argument("--output", default=os.path.join("artifacts", "degradation"))

    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    if args.cmd == "hypothesize":
        run_hypothesize(args.phases, args.output)
    elif args.cmd == "all":
        run_all_degradation(args.demos, args.raw, args.phases, args.fcm, args.output, args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
