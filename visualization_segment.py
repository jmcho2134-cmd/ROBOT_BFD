#!/usr/bin/env python
"""visualize_segmentation.py — M1(phase_segment) 결과 시각화.

세 가지를 만든다:
  1) seg_phases_3d.png / .mp4  : eef 궤적을 phase 로 색칠한 3D 플롯(+회전 영상)
  2) seg_subgoals.png          : phase 별 subgoal(main/change/hold) 정의 패널
  3) phase_{i}_{name}.mp4      : phase 별 영상
        --mode traj (기본)  eef+object 3D 궤적 애니메이션 (robosuite 불필요, 어디서나)
        --mode sim          robosuite 로 실제 로봇 장면 오프스크린 렌더 (robosuite 필요)

입력(모두 레포 아티팩트):
  artifacts/segmentation/phase_seg_{tag}.npz   (z, bounds, labels, subgoal_set, F, goal)
  artifacts/segmentation/subgoals.json         (main/change/hold/passive/window, 있으면 우선)
  artifacts/raw/raw_{tag}.npz                  (eef_pos, object_pos, goal, gripper, contact)
  data/demos/**/demo.hdf5                       (--mode sim 일 때만)

사용:
  python visualize_segmentation.py                       # 전부(traj 영상)
  python visualize_segmentation.py --no-video            # 3D/subgoal PNG 만
  python visualize_segmentation.py --mode sim            # 실제 로봇 렌더(robosuite)
"""
import os
import json
import argparse
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless 저장용
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# phase 색(질적 팔레트; 최대 8 phase)
PHASE_COLORS = ["#2563eb", "#16a34a", "#ea580c", "#9333ea",
                "#dc2626", "#0891b2", "#ca8a04", "#4b5563"]


# ---------------------------------------------------------------------------
# 로딩
# ---------------------------------------------------------------------------
def discover_tag(seg_dir):
    paths = sorted(glob(os.path.join(seg_dir, "phase_seg_*.npz")))
    if not paths:
        raise SystemExit(f"{seg_dir} 아래 phase_seg_*.npz 없음. phase_segment 먼저 실행.")
    return [os.path.basename(p)[len("phase_seg_"):-len(".npz")] for p in paths]


def load_segmentation(seg_dir, tag):
    seg = np.load(os.path.join(seg_dir, f"phase_seg_{tag}.npz"), allow_pickle=True)
    z = seg["z"].astype(int)
    labels = [str(x) for x in seg["labels"]]
    bounds = [int(x) for x in seg["bounds"]]
    goal = np.asarray(seg["goal"], float) if "goal" in seg.files else None
    # subgoal: subgoals.json 우선, 없으면 npz 의 subgoal_set
    subgoals = {}
    jpath = os.path.join(seg_dir, "subgoals.json")
    if os.path.exists(jpath):
        js = json.load(open(jpath))
        if tag in js:
            subgoals = {int(k): v for k, v in js[tag].items()}
    if not subgoals and "subgoal_set" in seg.files:
        subgoals = {int(k): v for k, v in seg["subgoal_set"]}
    return dict(z=z, labels=labels, bounds=bounds, goal=goal, subgoals=subgoals)


def load_frames(raw_dir, tag, goal_fallback=None):
    raw = np.load(os.path.join(raw_dir, f"raw_{tag}.npz"), allow_pickle=True)
    eef = np.asarray(raw["eef_pos"], float)
    obj = np.asarray(raw["object_pos"], float)
    goal = np.asarray(raw["goal"], float) if "goal" in raw.files else goal_fallback
    grip = np.asarray(raw["gripper_aperture"], float) if "gripper_aperture" in raw.files else None
    return dict(eef=eef, obj=obj, goal=goal, grip=grip)


def phase_windows(z):
    """각 phase 의 [start, end) 리스트. z 는 오름차순 phase 라벨 가정."""
    out = []
    for ph in sorted(np.unique(z)):
        idx = np.where(z == ph)[0]
        out.append((int(ph), int(idx[0]), int(idx[-1]) + 1))
    return out


def pcolor(ph):
    return PHASE_COLORS[int(ph) % len(PHASE_COLORS)]


# ---------------------------------------------------------------------------
# 1) 3D phase 플롯
# ---------------------------------------------------------------------------
def _equal_aspect_3d(ax, pts):
    """3D 축 비율을 데이터에 맞게 균등화."""
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    ctr = (mn + mx) / 2.0
    r = float((mx - mn).max()) / 2.0 + 1e-6
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)


def plot_phases_3d(eef, obj, z, goal, labels, out_png, view=(22, -60)):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # eef 궤적을 point 별 phase 색으로 (Line3DCollection)
    segs = np.stack([eef[:-1], eef[1:]], axis=1)
    seg_colors = [pcolor(z[i]) for i in range(len(eef) - 1)]
    lc = Line3DCollection(segs, colors=seg_colors, linewidths=3.0)
    ax.add_collection3d(lc)

    # object 궤적(옅은 점선)
    ax.plot(obj[:, 0], obj[:, 1], obj[:, 2], color="#9ca3af",
            ls="--", lw=1.2, alpha=0.8, label="object path")

    # phase 경계 지점(색 큰 점)
    for ph, a, b in phase_windows(z):
        ax.scatter(*eef[a], color=pcolor(ph), s=70, edgecolors="k",
                   linewidths=0.6, zorder=5)

    # start / goal
    ax.scatter(*eef[0], color="k", s=90, marker="o", label="eef start", zorder=6)
    if goal is not None:
        ax.scatter(*goal, color="#eab308", s=260, marker="*",
                   edgecolors="k", linewidths=0.8, label="goal", zorder=6)

    _equal_aspect_3d(ax, np.vstack([eef, obj]))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_title("Phase segmentation — end-effector trajectory (colored by phase)")

    # 범례: phase 색 + 마커
    handles = [Line2D([0], [0], color=pcolor(ph), lw=4,
                      label=f"P{ph}: {labels[ph] if ph < len(labels) else ph}")
               for ph, _, _ in phase_windows(z)]
    handles += [Line2D([0], [0], color="#9ca3af", ls="--", lw=1.5, label="object path"),
                Line2D([0], [0], color="k", marker="o", ls="", label="eef start"),
                Line2D([0], [0], color="#eab308", marker="*", ls="", ms=12, label="goal")]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def animate_orbit_3d(eef, obj, z, goal, labels, out_path, fps=20, n_frames=120):
    """3D phase 플롯을 회전시키는 영상."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    segs = np.stack([eef[:-1], eef[1:]], axis=1)
    lc = Line3DCollection(segs, colors=[pcolor(z[i]) for i in range(len(eef) - 1)],
                          linewidths=3.0)
    ax.add_collection3d(lc)
    ax.plot(obj[:, 0], obj[:, 1], obj[:, 2], color="#9ca3af", ls="--", lw=1.2, alpha=0.8)
    ax.scatter(*eef[0], color="k", s=90, zorder=6)
    if goal is not None:
        ax.scatter(*goal, color="#eab308", s=260, marker="*",
                   edgecolors="k", zorder=6)
    _equal_aspect_3d(ax, np.vstack([eef, obj]))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Phase segmentation (orbit view)")

    def update(i):
        ax.view_init(elev=22, azim=-60 + 360 * i / n_frames)
        return ()

    _save_anim(FuncAnimation(fig, update, frames=n_frames, blit=False),
               out_path, fps)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2) subgoal 정의 패널
# ---------------------------------------------------------------------------
def _arrow(sign):
    return "↑" if float(sign) > 0 else ("↓" if float(sign) < 0 else "·")


def plot_subgoals(subgoals, labels, windows, out_png):
    phases = sorted(subgoals.keys())
    n = len(phases)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 6.2))
    if n == 1:
        axes = [axes]

    for ax, ph in zip(axes, phases):
        sg = subgoals[ph]
        ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        col = pcolor(ph)
        # 헤더 바
        ax.add_patch(plt.Rectangle((0, 0.92), 1, 0.08, color=col, transform=ax.transAxes))
        lbl = labels[ph] if ph < len(labels) else str(ph)
        ax.text(0.5, 0.96, f"P{ph}", ha="center", va="center",
                color="white", fontsize=13, fontweight="bold", transform=ax.transAxes)
        win = None
        if isinstance(sg, dict) and "window" in sg:
            win = sg["window"]
        elif ph < len(windows):
            win = [windows[ph][1], windows[ph][2]]
        sub = lbl + (f"\n[t {win[0]}–{win[1]}]" if win else "")
        ax.text(0.5, 0.86, sub, ha="center", va="top", fontsize=8.5,
                color="#374151", transform=ax.transAxes)

        y = 0.74

        def section(title, y):
            ax.text(0.04, y, title, fontsize=9.5, fontweight="bold",
                    color=col, transform=ax.transAxes)
            return y - 0.055

        # MAIN
        y = section("MAIN (boundary)", y)
        main = sg.get("main", {}) if isinstance(sg, dict) else {}
        if main:
            ax.text(0.09, y, f"{main.get('feature','?')}  "
                             f"degrade {_arrow(main.get('degrade_sign', 0))}",
                    fontsize=9, transform=ax.transAxes)
            y -= 0.05
        y -= 0.02

        # CHANGE
        y = section("CHANGE", y)
        for c in (sg.get("change", []) if isinstance(sg, dict) else []):
            ax.text(0.09, y, f"{c.get('feature','?')}  {_arrow(c.get('degrade_sign',0))}",
                    fontsize=8.5, transform=ax.transAxes)
            y -= 0.045
        y -= 0.02

        # HOLD
        y = section("HOLD (keep stable)", y)
        for h in (sg.get("hold", []) if isinstance(sg, dict) else []):
            ax.text(0.09, y, f"{h}", fontsize=8.5, transform=ax.transAxes)
            y -= 0.045
        y -= 0.02

        # PASSIVE 는 개수만
        passive = sg.get("passive", []) if isinstance(sg, dict) else []
        if passive:
            ax.text(0.04, max(y, 0.03), f"passive: {len(passive)} feats (no role)",
                    fontsize=7.5, color="#9ca3af", style="italic", transform=ax.transAxes)

    fig.suptitle("Phase-local subgoals  (MAIN=boundary feature, CHANGE=varying, "
                 "HOLD=kept stable, arrow=degrade direction)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# 3) phase 별 영상 (traj 모드)
# ---------------------------------------------------------------------------
def _save_anim(anim, out_path, fps):
    if out_path.lower().endswith(".mp4"):
        try:
            anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2400))
            return
        except Exception as ex:
            print(f"[warn] ffmpeg 저장 실패({ex}) -> gif 로 대체")
            out_path = out_path[:-4] + ".gif"
    anim.save(out_path, writer=PillowWriter(fps=fps))


def animate_phase_traj(eef, obj, z, goal, ph, label, out_path, fps=20, max_frames=140):
    """해당 phase 구간의 eef+object 3D 궤적 애니메이션."""
    idx = np.where(z == ph)[0]
    a, b = int(idx[0]), int(idx[-1]) + 1
    step = max(1, (b - a) // max_frames)
    fr = list(range(a, b, step))
    col = pcolor(ph)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    # 전체 궤적을 옅게 배경으로
    ax.plot(eef[:, 0], eef[:, 1], eef[:, 2], color="#d1d5db", lw=0.8, alpha=0.6)
    ax.plot(obj[:, 0], obj[:, 1], obj[:, 2], color="#e5e7eb", ls="--", lw=0.8, alpha=0.6)
    if goal is not None:
        ax.scatter(*goal, color="#eab308", s=220, marker="*", edgecolors="k", zorder=6)
    _equal_aspect_3d(ax, np.vstack([eef, obj]))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    (trail,) = ax.plot([], [], [], color=col, lw=3.0, label="eef (this phase)")
    eef_pt = ax.scatter([], [], [], color=col, s=80, edgecolors="k", zorder=7)
    obj_pt = ax.scatter([], [], [], color="#111827", s=60, marker="s", zorder=7)
    title = ax.set_title("")

    def update(k):
        t = fr[k]
        trail.set_data(eef[a:t + 1, 0], eef[a:t + 1, 1])
        trail.set_3d_properties(eef[a:t + 1, 2])
        eef_pt._offsets3d = ([eef[t, 0]], [eef[t, 1]], [eef[t, 2]])
        obj_pt._offsets3d = ([obj[t, 0]], [obj[t, 1]], [obj[t, 2]])
        title.set_text(f"P{ph} {label}   frame {t}/{b}")
        return ()

    ax.view_init(elev=22, azim=-60)
    _save_anim(FuncAnimation(fig, update, frames=len(fr), blit=False), out_path, fps)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3') phase 별 영상 (sim 모드; robosuite 필요)
# ---------------------------------------------------------------------------
def render_phase_sim(demo_path, z, out_dir, tag, labels, fps=20,
                     cam="frontview", W=640, H=480, max_frames=160):
    """robosuite 로 데모 state 를 복원하며 phase 구간을 오프스크린 렌더 -> phase 별 mp4.
    레포의 phase_segment 헬퍼(read_demo/build_env/reset_to_scene)를 재사용한다."""
    try:
        import imageio
        import phase_segment as ps
    except Exception as ex:
        raise SystemExit(f"[sim] robosuite/phase_segment import 실패: {ex}\n"
                         f"      --mode traj 로 대신 실행하세요.")
    env_info, demos = ps.read_demo(demo_path)
    env = ps.build_env(env_info)
    outs = []
    try:
        for name, states, actions, model_xml in demos:
            ps.reset_to_scene(env, model_xml)
            for ph, a, b in phase_windows(z):
                step = max(1, (b - a) // max_frames)
                frames = []
                for t in range(a, min(b, len(states)), step):
                    env.sim.set_state_from_flattened(states[t]); env.sim.forward()
                    img = env.sim.render(width=W, height=H, camera_name=cam)[::-1]
                    frames.append(img)
                lbl = labels[ph] if ph < len(labels) else str(ph)
                out = os.path.join(out_dir, f"phase_{ph}_{_safe(lbl)}_sim.mp4")
                imageio.mimsave(out, frames, fps=fps)
                outs.append(out)
                print(f"  [sim] P{ph} -> {out}  ({len(frames)} frames)")
            break                                   # 첫 데모만
    finally:
        try:
            env.close()
        except Exception:
            pass
    return outs


# ---------------------------------------------------------------------------
def _safe(s):
    return "".join(c if c.isalnum() else "_" for c in s)[:24]


def main():
    ap = argparse.ArgumentParser(description="M1 세그멘테이션 시각화")
    ap.add_argument("--seg", default=os.path.join("artifacts", "segmentation"))
    ap.add_argument("--raw", default=os.path.join("artifacts", "raw"))
    ap.add_argument("--demos", default=os.path.join("data", "demos"),
                    help="--mode sim 일 때 demo.hdf5 검색 루트")
    ap.add_argument("--pattern", default="demo.hdf5")
    ap.add_argument("--out", default=os.path.join("artifacts", "segmentation", "viz"))
    ap.add_argument("--tag", default=None, help="비우면 phase_seg_*.npz 전체")
    ap.add_argument("--mode", choices=["traj", "sim"], default="traj",
                    help="phase 영상: traj=3D 궤적 애니메이션, sim=robosuite 렌더")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--no-video", action="store_true", help="영상 없이 PNG 만")
    ap.add_argument("--cam", default="frontview", help="sim 카메라 이름")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tags = [args.tag] if args.tag else discover_tag(args.seg)

    for tag in tags:
        print(f"\n=== {tag} ===")
        seg = load_segmentation(args.seg, tag)
        fr = load_frames(args.raw, tag, goal_fallback=seg["goal"])
        z, labels = seg["z"], seg["labels"]
        goal = fr["goal"] if fr["goal"] is not None else seg["goal"]
        wins = phase_windows(z)
        odir = os.path.join(args.out, tag)
        os.makedirs(odir, exist_ok=True)

        # 1) 3D phase 플롯
        p1 = plot_phases_3d(fr["eef"], fr["obj"], z, goal, labels,
                            os.path.join(odir, "seg_phases_3d.png"))
        print(f"  [1] 3D phase 플롯 -> {p1}")

        # 2) subgoal 패널
        p2 = plot_subgoals(seg["subgoals"], labels, wins,
                           os.path.join(odir, "seg_subgoals.png"))
        print(f"  [2] subgoal 패널 -> {p2}")

        if args.no_video:
            continue

        # 1') 회전 영상
        p3 = animate_orbit_3d(fr["eef"], fr["obj"], z, goal, labels,
                              os.path.join(odir, "seg_phases_3d.mp4"), fps=args.fps)
        print(f"  [1'] 3D 회전 영상 -> {p3}")

        # 3) phase 별 영상
        if args.mode == "sim":
            demo_paths = sorted(glob(os.path.join(args.demos, "**", args.pattern),
                                     recursive=True))
            dmap = {os.path.basename(os.path.dirname(p)): p for p in demo_paths}
            dp = dmap.get(tag) or (demo_paths[0] if demo_paths else None)
            if not dp:
                print("  [3] demo.hdf5 없음 -> sim 스킵"); continue
            render_phase_sim(dp, z, odir, tag, labels, fps=args.fps, cam=args.cam)
        else:
            for ph, a, b in wins:
                lbl = labels[ph] if ph < len(labels) else str(ph)
                out = os.path.join(odir, f"phase_{ph}_{_safe(lbl)}.mp4")
                animate_phase_traj(fr["eef"], fr["obj"], z, goal, ph, lbl, out, fps=args.fps)
                print(f"  [3] P{ph} 영상 -> {out}")

    print("\n완료.")


if __name__ == "__main__":
    main()
