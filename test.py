#!/usr/bin/env python
"""test_noise_trajectories.py — baseline 노이즈가 데모 궤적을 어떻게 바꾸는지 레벨별 시각화.

두 노이즈를 orig/mild/medium/severe 4단계로 실제 시뮬레이터에 굴려 나란히 비교한다.

  A) Gaussian (전방향, 스텝마다 독립):  a'_t = clip(a_t + N(0, σ_level))    ← D-REX/SSRR 식
  B) Fixed unit direction (고정 1방향):  a'_t = clip(a_t + λ_level · d)      ← d 는 모든 t 동일

둘 다 레포 실제 파이프라인과 동일하게 fs.perturb_action(clip + proper rotation)을 쓴다.

출력(artifacts/segmentation/viz/{tag}/noise/):
  noise_traj_3d.png        : 두 방식의 3D eef 궤적 (레벨 색). orig=검정.
  noise_success.png        : 레벨별 task success / final_goal_dist 요약
  noise_{gauss,fixed}.mp4  : (옵션) 레벨별 궤적이 함께 발산하는 애니메이션

기본: robosuite 실제 롤아웃(재민 실행).  --selftest: mock 물리로 파이프라인 검증(robosuite 불필요).

사용:
  python test_noise_trajectories.py                          # 실제 롤아웃
  python test_noise_trajectories.py --levels 0 0.05 0.1 0.2  # 레벨 직접 지정
  python test_noise_trajectories.py --video                  # 영상까지
  python test_noise_trajectories.py --selftest               # robosuite 없이 검증
"""
import os
import json
import argparse
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401 (parity with viz)

import feature_select as fs
import phase_segment as ps
import degradation as deg   # TrajectoryExecutor, _MockExecutor 재사용

LEVEL_NAMES = ["orig", "mild", "medium", "severe"]
LEVEL_COLORS = ["#111827", "#f59e0b", "#ea580c", "#dc2626"]  # orig=검정, 이후 warm gradient


# ---------------------------------------------------------------------------
# 노이즈 적용 (둘 다 레포의 clip+회전합성 사용)
# ---------------------------------------------------------------------------
def gaussian_noise_actions(actions, sigma, rng, adim):
    """전방향, 스텝마다 독립 가우시안: a'_t = clip(a_t + N(0, σ))."""
    A = np.array(actions, float).copy()
    if sigma <= 0:
        return A
    for t in range(len(A)):
        d = rng.normal(0.0, sigma, adim)
        A[t] = fs.perturb_action(A[t], d, adim, proper_rotation=True)
    return A


def fixed_direction_actions(actions, direction, lam, adim):
    """고정 단위방향: a'_t = clip(a_t + λ·d), 모든 t 동일."""
    A = np.array(actions, float).copy()
    if lam <= 0:
        return A
    d = np.asarray(direction, float)
    for t in range(len(A)):
        A[t] = fs.perturb_action(A[t], lam * d, adim, proper_rotation=True)
    return A


def random_unit_direction(adim, rng):
    v = rng.normal(0.0, 1.0, adim)
    return v / (np.linalg.norm(v) + 1e-12)


# ---------------------------------------------------------------------------
# 롤아웃: (type, level) 마다 실행 -> 궤적 + success + final_goal_dist
# ---------------------------------------------------------------------------
def rollout_levels(execu, actions, levels, adim, rng, fixed_dir):
    """반환 dict: 'gauss'/'fixed' 각각 레벨별 [dict(eef,obj,success,fgd,ok,level)]."""
    out = {"gauss": [], "fixed": []}
    for lv in levels:
        Ag = gaussian_noise_actions(actions, lv, rng, adim)
        fg, ig = execu.execute(Ag)
        out["gauss"].append(dict(eef=fg["eef"], obj=fg["obj"], level=float(lv),
                                 success=ig.get("success"), fgd=ig.get("final_goal_dist"),
                                 ok=ig.get("ok", True)))
        Af = fixed_direction_actions(actions, fixed_dir, lv, adim)
        ff, if_ = execu.execute(Af)
        out["fixed"].append(dict(eef=ff["eef"], obj=ff["obj"], level=float(lv),
                                 success=if_.get("success"), fgd=if_.get("final_goal_dist"),
                                 ok=if_.get("ok", True)))
    return out


# ---------------------------------------------------------------------------
# 시각화 1: 3D 궤적 비교
# ---------------------------------------------------------------------------
def _equal_aspect_3d(ax, pts):
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    ctr = (mn + mx) / 2.0
    r = float((mx - mn).max()) / 2.0 + 1e-6
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)


def _succ_mark(s):
    return "✓" if s is True else ("✗" if s is False else "?")


def plot_noise_3d(rolls, goal, levels, out_png, titles=("Gaussian (per-step, all-dim)",
                                                        "Fixed unit direction")):
    fig = plt.figure(figsize=(14, 6.5))
    allpts = np.vstack([r["eef"] for k in rolls for r in rolls[k]])
    for col, (key, title) in enumerate(zip(["gauss", "fixed"], titles)):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        for i, r in enumerate(rolls[key]):
            c = LEVEL_COLORS[i % len(LEVEL_COLORS)]
            lw = 3.0 if i == 0 else 2.0
            ax.plot(r["eef"][:, 0], r["eef"][:, 1], r["eef"][:, 2],
                    color=c, lw=lw, alpha=0.95)
        if goal is not None:
            ax.scatter(*goal, color="#eab308", s=240, marker="*",
                       edgecolors="k", zorder=6)
        ax.scatter(*rolls[key][0]["eef"][0], color="k", s=70, zorder=6)
        _equal_aspect_3d(ax, allpts)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=22, azim=-60)
        ax.set_title(title)
        handles = [Line2D([0], [0], color=LEVEL_COLORS[i % len(LEVEL_COLORS)], lw=3,
                          label=f"{LEVEL_NAMES[i] if i < len(LEVEL_NAMES) else i} "
                                f"(σ/λ={r['level']:.2f}) succ={_succ_mark(r['success'])}")
                   for i, r in enumerate(rolls[key])]
        handles.append(Line2D([0], [0], color="#eab308", marker="*", ls="", ms=12,
                              label="goal"))
        ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9)
    fig.suptitle("Noise on demo actions — eef trajectory by level  "
                 "(orig=black; warmer=stronger noise)", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# 시각화 2: success / final_goal_dist 요약
# ---------------------------------------------------------------------------
def plot_noise_success(rolls, levels, out_png, succ_thresh=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(levels))
    for key, color, mk in [("gauss", "#2563eb", "o"), ("fixed", "#dc2626", "s")]:
        fgd = [r["fgd"] if r["fgd"] is not None else np.nan for r in rolls[key]]
        ax.plot(x, fgd, color=color, marker=mk, lw=2, label=f"{key}: final_goal_dist")
        for i, r in enumerate(rolls[key]):
            if r["success"] is False:                       # 실패 지점 크게 표시
                ax.scatter(x[i], fgd[i], s=180, facecolors="none",
                           edgecolors=color, linewidths=2.2, zorder=5)
    if succ_thresh is not None:
        ax.axhline(succ_thresh, color="#6b7280", ls="--", lw=1,
                   label=f"success threshold ({succ_thresh})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{LEVEL_NAMES[i] if i < len(LEVEL_NAMES) else i}\n"
                        f"{levels[i]:.2f}" for i in range(len(levels))])
    ax.set_ylabel("final object–goal distance (m)")
    ax.set_xlabel("noise level (σ for gauss, λ for fixed)")
    ax.set_title("Does noise keep success? (open circle = task FAILED)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# 시각화 3: 레벨별 궤적 발산 애니메이션 (옵션)
# ---------------------------------------------------------------------------
def animate_noise(rolls, key, goal, out_path, fps=20, max_frames=140):
    trajs = rolls[key]
    T = max(len(r["eef"]) for r in trajs)
    step = max(1, T // max_frames)
    fr = list(range(0, T, step))
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    allpts = np.vstack([r["eef"] for r in trajs])
    if goal is not None:
        ax.scatter(*goal, color="#eab308", s=220, marker="*", edgecolors="k", zorder=6)
    _equal_aspect_3d(ax, allpts)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-60)
    lines = [ax.plot([], [], [], color=LEVEL_COLORS[i % len(LEVEL_COLORS)],
                     lw=3 if i == 0 else 2)[0] for i in range(len(trajs))]

    def update(k):
        t = fr[k]
        for ln, r in zip(lines, trajs):
            tt = min(t, len(r["eef"]) - 1)
            ln.set_data(r["eef"][:tt + 1, 0], r["eef"][:tt + 1, 1])
            ln.set_3d_properties(r["eef"][:tt + 1, 2])
        ax.set_title(f"{key} noise — trajectories diverging   frame {t}/{T}")
        return ()

    anim = FuncAnimation(fig, update, frames=len(fr), blit=False)
    if out_path.lower().endswith(".mp4"):
        try:
            anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2400))
        except Exception as ex:
            print(f"[warn] ffmpeg 실패({ex}) -> gif"); out_path = out_path[:-4] + ".gif"
            anim.save(out_path, writer=PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def _load_goal(seg_dir, raw_dir, tag):
    for d, key in [(raw_dir, "raw"), (seg_dir, "seg")]:
        p = os.path.join(raw_dir if key == "raw" else seg_dir,
                         (f"raw_{tag}.npz" if key == "raw" else f"phase_seg_{tag}.npz"))
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            if "goal" in z.files:
                return np.asarray(z["goal"], float)
    return None


def run_real(args, tag):
    demo_paths = sorted(glob(os.path.join(args.demos, "**", args.pattern), recursive=True))
    dmap = {os.path.basename(os.path.dirname(p)): p for p in demo_paths}
    dp = dmap.get(tag) or (demo_paths[0] if demo_paths else None)
    if not dp:
        raise SystemExit(f"{args.demos} 아래 {args.pattern} 없음.")
    env_info, demos = ps.read_demo(dp)
    env = ps.build_env(env_info)
    object_type = env_info.get("object_type", None)
    raw = np.load(os.path.join(args.raw, f"raw_{tag}.npz"), allow_pickle=True)
    dt = float(raw["dt"]) if "dt" in raw.files else 0.05
    goal = _load_goal(args.seg, args.raw, tag)
    rng = np.random.default_rng(args.seed)
    try:
        name, states, actions, model_xml = demos[0]
        ps.reset_to_scene(env, model_xml)
        obj0_z = float(raw["object_pos"][0, 2]) if "object_pos" in raw.files else 0.85
        execu = deg.TrajectoryExecutor(env, states, object_type, dt, goal, obj0_z)
        fixed_dir = random_unit_direction(actions.shape[1], rng)
        rolls = rollout_levels(execu, actions, args.levels, actions.shape[1], rng, fixed_dir)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return rolls, goal


def run_selftest(args):
    """robosuite 없이 mock 물리로 파이프라인 검증."""
    print("=== noise test SELFTEST (mock physics; no robosuite) ===")
    T, adim = 240, 7
    actions = np.zeros((T, adim)); actions[:, 0] = 0.25          # +x 로 이동
    actions[80:, 6] = 1.0                                        # 중간부터 grip 닫음
    execu = deg._MockExecutor(T=T, adim=adim); goal = execu.goal
    rng = np.random.default_rng(0)
    fixed_dir = random_unit_direction(adim, rng)
    rolls = rollout_levels(execu, actions, args.levels, adim, rng, fixed_dir)
    return rolls, goal


def main():
    ap = argparse.ArgumentParser(description="baseline 노이즈 궤적 레벨별 시각화")
    ap.add_argument("--seg", default=os.path.join("artifacts", "segmentation"))
    ap.add_argument("--raw", default=os.path.join("artifacts", "raw"))
    ap.add_argument("--demos", default=os.path.join("data", "demos"))
    ap.add_argument("--pattern", default="demo.hdf5")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=os.path.join("artifacts", "segmentation", "viz"))
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4],
                    help="노이즈 크기 4단계(σ=gauss, λ=fixed). 첫 값은 보통 0(orig).")
    ap.add_argument("--succ-thresh", type=float, default=None,
                    help="success 판정 거리(요약 그림 참고선). 없으면 생략")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true", help="레벨 발산 애니메이션도 저장")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        rolls, goal = run_selftest(args)
        odir = os.path.join(args.out, "_selftest_noise")
        tags = ["_selftest"]
    else:
        tag = args.tag
        if tag is None:
            segp = sorted(glob(os.path.join(args.seg, "phase_seg_*.npz")))
            if not segp:
                raise SystemExit("phase_seg_*.npz 없음.")
            tag = os.path.basename(segp[0])[len("phase_seg_"):-len(".npz")]
        rolls, goal = run_real(args, tag)
        odir = os.path.join(args.out, tag, "noise")
        tags = [tag]

    os.makedirs(odir, exist_ok=True)
    p1 = plot_noise_3d(rolls, goal, args.levels, os.path.join(odir, "noise_traj_3d.png"))
    print(f"[1] 3D 궤적 비교 -> {p1}")
    p2 = plot_noise_success(rolls, args.levels, os.path.join(odir, "noise_success.png"),
                            succ_thresh=args.succ_thresh)
    print(f"[2] success 요약 -> {p2}")

    # 콘솔 요약
    print("\n level  |  gauss(succ, fgd)      |  fixed(succ, fgd)")
    for i, lv in enumerate(args.levels):
        g, f = rolls["gauss"][i], rolls["fixed"][i]
        print(f"  {lv:<5.2f} |  {_succ_mark(g['success'])}  "
              f"{('%.3f'%g['fgd']) if g['fgd'] is not None else '   -'}"
              f"            |  {_succ_mark(f['success'])}  "
              f"{('%.3f'%f['fgd']) if f['fgd'] is not None else '   -'}")

    if args.video:
        for key in ["gauss", "fixed"]:
            vp = animate_noise(rolls, key, goal, os.path.join(odir, f"noise_{key}.mp4"))
            print(f"[3] {key} 애니메이션 -> {vp}")
    print("\n완료.")


if __name__ == "__main__":
    main()
