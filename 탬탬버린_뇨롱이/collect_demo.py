#!/usr/bin/env python
"""
collect_demo.py
===============

Interactive keyboard-teleop demo collector for **any robosuite 1.5 environment
and robot**, generalized from this project's ``collect_pickplace_can.py``.

What changed vs. collect_pickplace_can.py
-----------------------------------------
``collect_pickplace_can.py`` was hard-wired to *PickPlaceCan + Panda* with an
enforced *OSC_POSITION* (4-dim, position-only) action space, so the arm could
not roll and the scene/robot could not be changed. This script instead follows
robosuite's own reference scripts:

* ``robosuite/demos/demo_random_action.py`` -> **terminal menus** to pick the
  environment, robot(s) (and the two-arm configuration) via
  ``choose_environment`` / ``choose_robots`` / ``choose_multi_arm_config`` from
  ``robosuite.utils.input_utils``.
* ``robosuite/scripts/collect_human_demonstrations.py`` -> the actual teleop
  collection loop (``collect_human_trajectory``), reused verbatim. Its HDF5
  writer is intentionally NOT reused: we save with our own ``save_episode_as_hdf5``
  so an accepted demo is written whether or not the env auto-flagged success
  (the same layout robosuite's writer produces).

Two things the user explicitly asked for
-----------------------------------------
1. **The camera moves.** Default renderer is ``mjviewer`` (the native MuJoCo
   passive viewer): drag with the mouse to orbit / pan / zoom the camera
   freely while teleoperating. ``--renderer mujoco`` (the OpenCV viewer with a
   fixed *named* ``--camera``) is still available.
2. **The arm can roll (rotate).** Default arm controller is **OSC_POSE**
   (6-DOF pose -> 7-dim action). With OSC_POSE the stock ``Keyboard`` device's
   rotation keys (e/r/y/h/p/o) are live, so the end-effector can roll/pitch/yaw.
   ``--controller OSC_POSITION`` restores the 4-dim, position-only behaviour
   that the downstream feature-bank / reward-net pipeline expects (rotation
   disabled) via the ``OSCPositionKeyboard`` shim kept from the original file.

Nothing under the robosuite install is modified; all customization lives here.

Usage (activate the conda ``robosuite`` env first)::

    python collect_demo.py                       # fully interactive menus
    python collect_demo.py --environment Lift --robots Panda
    python collect_demo.py --controller OSC_POSITION   # 4-dim, no roll (pipeline)
    python collect_demo.py --renderer mujoco --camera agentview

Quit: press ``q`` in the keyboard listener to end the current episode. After
each episode the terminal asks whether to save it (``y`` -> write demo_<N>,
``n`` -> discard and reset to re-collect). Press ``Ctrl+C`` in the terminal to
stop the whole program.
"""

import argparse
import datetime
import hashlib
import inspect
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

# 다운스트림 M1/M2 와 스키마/replay 를 공유한다(중복 구현 금지, blueprint Sec 1.1).
#   feature_select(fs): raw npz 스키마 버전 + feature 계산 함수.
#   phase_segment(ps) : robosuite replay helper(read_demo/build_env/reset_to_scene/
#                       replay/resolve_object/...). extract 가 이걸 그대로 재사용한다.
# 두 모듈 다 robosuite 를 함수 안에서 지연 import 하므로 이 import 자체는 robosuite 없이도
# 로드된다 -> inspect/extract/selftest 가 teleop 스택 없이 동작.
import feature_select as fs
import phase_segment as ps

# robosuite teleop 스택(controllers/devices/scripts/wrappers/input_utils)은 무겁고
# pynput/디스플레이를 요구하므로 top-level 에서 import 하지 않는다. 실제 수집 경로
# (run_collect / 아래 함수들)가 호출될 때 함수 내부에서 지연 import 한다. 그래야 이 파일이
# robosuite teleop 스택 없이도 inspect/extract/selftest 로 실행된다.


# ---------------------------------------------------------------------------
# Custom keyboard device factory: position-only (OSC_POSITION) teleoperation
# ---------------------------------------------------------------------------
def _build_teleop_device(args, env):
    """teleop 디바이스를 만든다. robosuite.devices.Keyboard 를 지연 import 한다(그래야
    inspect/extract/selftest 는 pynput/디스플레이 없이 동작). OSC_POSITION 모드에서만
    position-only shim 을 정의해 반환하고, OSC_POSE 는 stock Keyboard 를 그대로 쓴다(회전키
    활성 -> 팔 롤링). 동작/스케일링은 기존 OSCPositionKeyboard 와 동일하게 유지한다."""
    from robosuite.devices import Keyboard

    if args.controller != "OSC_POSITION":
        # OSC_POSE: stock Keyboard 그대로(회전키 e/r/y/h/p/o 활성).
        return Keyboard(env=env, pos_sensitivity=args.pos_sensitivity,
                        rot_sensitivity=args.rot_sensitivity)

    class OSCPositionKeyboard(Keyboard):
        """3-DOF position-only arm delta 를 내보내는 Keyboard.

        stock Device.input2action(arm 컨트롤러가 OSC_POSE/JOINT_POSITION 이라고 assert
        하고 6-DOF delta 를 만듦)을 우회해 OSC_POSITION arm 을 구동한다. 회전키
        (e/r/y/h/p/o)는 이 모드에서 무시된다(팔 롤링 없음). 나머지(pynput 리스너, 키
        바인딩, 그리퍼 토글, 리셋키)는 Keyboard 에서 상속. 다운스트림 4-dim 파이프라인과의
        바이트 호환을 위해 원본과 동일하게 유지한다."""

        # 1.5.2 는 goal_update_mode kwarg 를, 1.5.1 은 kwarg 없이 호출한다 -> 둘 다 수용.
        def input2action(self, mirror_actions=False, goal_update_mode="target"):
            robot = self.env.robots[self.active_robot]
            active_arm = self.active_arm

            state = self.get_controller_state()
            dpos = state["dpos"]
            raw_drotation = state["raw_drotation"]
            grasp = state["grasp"]
            reset = state["reset"]

            # 리셋('q' 키)은 None 반환 -> 에피소드 종료.
            if reset:
                return None

            # stock 디바이스와 "감각"을 맞추기 위한 per-teleop 스케일링 재현: dpos 는
            # _postprocess_device_outputs 안에서 (*75) 스케일 + [-1,1] 클립. drotation 은
            # 계산 후 버린다.
            drotation = raw_drotation[[1, 0, 2]]
            drotation[2] = -drotation[2]
            dpos, drotation = self._postprocess_device_outputs(dpos, drotation)

            # 그리퍼 토글 -> +1(닫힘)/-1(열림), stock 디바이스와 동일.
            grasp = 1 if grasp else -1

            ac_dict = {}
            # create_action_vector 가 제어되는 모든 arm part 에 대한 항목을 갖도록 0 delta
            # 로 채우고, active arm 만 아래에서 덮어쓴다.
            for arm in robot.arms:
                ctrl = robot.part_controllers[arm]
                assert ctrl.name == "OSC_POSITION", (
                    "OSCPositionKeyboard only supports OSC_POSITION arms; got "
                    f"'{ctrl.name}' for arm '{arm}'. Use the stock Keyboard for OSC_POSE."
                )
                ac_dict[f"{arm}_delta"] = np.zeros(ctrl.control_dim)  # control_dim == 3
                ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)

            # active arm: 3-DOF 위치 delta 만(회전 드롭), [-1,1] 클립.
            active_ctrl = robot.part_controllers[active_arm]
            ac_dict[f"{active_arm}_delta"] = np.clip(
                dpos[: active_ctrl.control_dim], -1.0, 1.0)

            gripper = robot.gripper[active_arm]
            if hasattr(gripper, "grasp_qpos"):
                ac_dict[f"{active_arm}_gripper"] = getattr(gripper, "grasp_qpos")[grasp]
            else:
                ac_dict[f"{active_arm}_gripper"] = np.array([grasp] * gripper.dof)

            return ac_dict

    return OSCPositionKeyboard(env=env, pos_sensitivity=args.pos_sensitivity,
                               rot_sensitivity=args.rot_sensitivity)


# ---------------------------------------------------------------------------
# Controller config: choose the arm controller (OSC_POSE default -> rolling)
# ---------------------------------------------------------------------------
def make_arm_controller_config(arm_controller, robot):
    """Return a BASIC composite-controller config with the arm part(s) set to
    ``arm_controller``.

    * ``OSC_POSE``      -> 6-DOF pose command -> 7-dim action (roll enabled).
    * ``OSC_POSITION``  -> 3-DOF position command -> 4-dim action (no roll).

    robosuite 1.5's ``load_composite_controller_config`` FLATTENS
    ``body_parts.arms`` into ``body_parts['right'] / ['left']`` (there is no
    'arms' key in the returned dict). We iterate those arm parts and swap the
    type. For OSC_POSITION we must ALSO resize output_max/output_min to length 3,
    because the BASIC (OSC_POSE) arm keeps length-6 output arrays and robosuite's
    ``nums2array`` does NOT truncate them to control_dim -> a shape mismatch
    during action scaling if left at length 6.
    """
    # 1.5 API: load_composite_controller_config replaces deprecated
    # load_controller_config (지연 import — teleop 경로에서만 필요).
    from robosuite.controllers import load_composite_controller_config

    # BASIC composite controller (generic). load_composite_controller_config
    # accepts a composite name ("BASIC") or a .json path -- NOT an arm-part name.
    config = load_composite_controller_config(controller="BASIC", robot=robot)

    if arm_controller == "OSC_POSITION":
        out_max, out_min = [0.05, 0.05, 0.05], [-0.05, -0.05, -0.05]
    elif arm_controller == "OSC_POSE":
        # Leave OSC_POSE at its BASIC defaults (length 6); nothing to resize.
        out_max = out_min = None
    else:
        out_max = out_min = None
        print(f"[warn] unrecognized arm controller '{arm_controller}'; not resizing output_max/min.")

    n_overridden = 0
    for part_name, part in config["body_parts"].items():
        # Only touch arm parts (they start out as OSC_POSE in BASIC).
        if isinstance(part, dict) and part.get("type") in ("OSC_POSE", "OSC_POSITION"):
            part["type"] = arm_controller
            if out_max is not None:
                part["output_max"] = list(out_max)
                part["output_min"] = list(out_min)
            n_overridden += 1

    if n_overridden == 0:
        print("[warn] no OSC arm parts found in BASIC config to override.")
    return config


# ---------------------------------------------------------------------------
# Terminal menus: environment / robot(s), mirroring demo_random_action.py
# ---------------------------------------------------------------------------
def choose_options_interactively(args):
    """Resolve env_name / robots / env_configuration.

    If the user passed --environment / --robots on the CLI we honour them;
    otherwise we fall back to robosuite's interactive terminal menus, following
    the exact branching used by robosuite/demos/demo_random_action.py so that
    two-arm and humanoid environments pick sensible robots.
    """
    # 지연 import — 대화형 메뉴/버전 배너는 teleop 수집 경로에서만 쓰인다.
    import robosuite as suite
    from robosuite.utils.input_utils import (
        choose_environment, choose_multi_arm_config, choose_robots)

    options = {}

    # print welcome info (same as demo_random_action.py)
    print("Welcome to robosuite v{}!".format(suite.__version__))
    if hasattr(suite, "__logo__"):
        print(suite.__logo__)

    # --- environment ---
    if args.environment is not None:
        options["env_name"] = args.environment
    else:
        options["env_name"] = choose_environment()

    # --- robot(s), with the multi-arm / humanoid branching from the demo ---
    if args.robots:
        # CLI override: one or more robot names given explicitly.
        options["robots"] = args.robots if len(args.robots) > 1 else args.robots[0]
        if "TwoArm" in options["env_name"] and args.env_configuration is not None:
            options["env_configuration"] = args.env_configuration
    elif "TwoArm" in options["env_name"]:
        # Choose env config and add it to options.
        options["env_configuration"] = (
            args.env_configuration if args.env_configuration is not None else choose_multi_arm_config()
        )
        # A bimanual config -> Baxter; otherwise the user picks two single arms.
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []
            print("A multiple single-arm configuration was chosen.\n")
            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))
    elif "Humanoid" in options["env_name"]:
        options["robots"] = choose_robots(use_humanoids=True)
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    return options


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------
def build_env(args, options, controller_config):
    """Create the (unwrapped) env, mirroring the stock collectors' suite.make.

    ``render_camera`` names the initial camera; with ``--renderer mjviewer`` you
    can then move that camera freely with the mouse. robosuite auto-enables the
    offscreen renderer only when needed, so has_offscreen_renderer=False is fine
    for on-screen teleop.
    """
    import robosuite as suite  # 지연 import (teleop 수집 경로 전용)

    # first robot name (str) -> used to load the composite controller defaults.
    env = suite.make(
        **options,
        controller_configs=controller_config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=args.control_freq,
    )

    # env.action_dim is None until the first reset(); env.action_spec is derived
    # from robot.action_limits and is valid immediately after make(). Use it.
    action_dim = int(env.action_spec[0].shape[0])
    arm_part = env.robots[0].part_controllers.get(args.arm)
    arm_controller_name = arm_part.name if arm_part is not None else "<unknown>"

    return env, action_dim, arm_controller_name


# ---------------------------------------------------------------------------
# Pretty launch banner
# ---------------------------------------------------------------------------
def print_launch_banner(args, options, out_dir, action_dim, arm_controller_name):
    rolling = arm_controller_name == "OSC_POSE"
    controls = [
        "  arrow keys  : move end-effector in x (up/down) and y (left/right)",
        "  . / ;       : move end-effector down / up (z)",
        "  spacebar    : toggle gripper open/close",
        "  q           : end the current episode (종료 후 저장 여부를 y/n 로 물어봄)",
    ]
    if rolling:
        controls.append("  e/r y/h p/o : ROLL / PITCH / YAW the end-effector  (OSC_POSE 회전키 활성)")
    else:
        controls.append("  (rotation keys e/r/y/h/p/o are IGNORED in OSC_POSITION mode -> 팔 롤링 없음)")

    print("\n" + "=" * 72)
    print(" robosuite keyboard demo collector  (환경/로봇 선택 + 카메라 이동 + 팔 롤링)")
    print("=" * 72)
    print(f" output directory : {out_dir}")
    print(f" environment      : {options['env_name']}")
    print(f" robot(s)         : {options.get('robots')}")
    if "env_configuration" in options:
        print(f" arm config       : {options['env_configuration']}")
    print(f" renderer / camera: {args.renderer} / {args.camera}")
    if args.renderer == "mjviewer":
        print("                    (마우스로 카메라 orbit/pan/zoom -> '카메라 이동' 가능)")
    print(f" controller       : {arm_controller_name}  ->  action_dim = {action_dim}")
    if rolling:
        print("                    (3 EE pos + 3 EE rot deltas + 1 gripper) -> 팔 롤링 O")
    elif action_dim == 4:
        print("                    (3 EE position deltas + 1 gripper) -> 팔 롤링 X, 4-dim 파이프라인 호환")
    print(f" control_freq     : {args.control_freq} Hz   (loop cap max_fr = {args.max_fr})")
    print(f" pos/rot sens.    : {args.pos_sensitivity} / {args.rot_sensitivity}")
    print("-" * 72)
    print(" keyboard controls:")
    for line in controls:
        print(line)
    print("-" * 72)
    print(" 키보드 리스너는 전역 pynput 훅이라 렌더 창에 포커스가 없어도 키가 잡힙니다.")
    print(" 프로그램 전체를 멈추려면 이 터미널에서 Ctrl+C 를 누르세요.")
    print(" 매 에피소드가 끝나면(q 또는 성공) 터미널에서 저장 여부(y/n)를 물어봅니다.")
    print("   y -> 저장 후 다음 데모 계속   /   n -> 저장 안 하고 리셋 후 재수집")
    print(f" 저장 위치: {os.path.join(out_dir, 'demo_<N>', 'demo.hdf5')}  (승인한 데모마다 번호별 폴더)")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Per-demo helpers: numbered demo_<N> folders + ask-to-save prompt
# (kept from collect_pickplace_can.py so each accepted demo lands in its own
#  demos/<Env>_<Robot>/demo_<N>/demo.hdf5)
# ---------------------------------------------------------------------------
def list_episode_dirs(tmp_directory):
    """Return the set of 'ep_*' episode sub-directory names in tmp_directory."""
    if not os.path.isdir(tmp_directory):
        return set()
    return {d for d in os.listdir(tmp_directory) if d.startswith("ep_")}


def newest_episode_dir(tmp_directory, before):
    """Return the full path of the ep_* dir created since the `before` snapshot.

    collect_human_trajectory() creates exactly one new episode directory per
    call (via DataCollectionWrapper._on_first_interaction), so the set
    difference against a pre-call snapshot pins down the just-collected episode.
    Returns None if nothing new was written (e.g. the user quit before taking a
    single step, so no interaction was ever logged).
    """
    new = sorted(list_episode_dirs(tmp_directory) - before)
    if not new:
        return None
    return os.path.join(tmp_directory, new[-1])


def episode_successful(ep_dir):
    """True if any state_*.npz in ep_dir recorded a successful=True flag.

    Mirrors gather_demonstrations_as_hdf5's own success criterion (OR of the
    per-flush `successful` flags) so what we ask about matches what would be
    written to the hdf5.
    """
    ok = False
    for state_file in glob(os.path.join(ep_dir, "state_*.npz")):
        dic = np.load(state_file, allow_pickle=True)
        ok = ok or bool(dic["successful"])
    return ok


def save_episode_as_hdf5(ep_dir, out_dir, env_info):
    """Write a single collected episode (ep_dir) into out_dir/demo.hdf5.

    This is a trimmed copy of robosuite's gather_demonstrations_as_hdf5 for ONE
    episode, with the crucial difference that it does NOT gate on the success
    flag: whatever the human accepted with 'y' is saved. The resulting file has
    the same layout robosuite produces (a single 'demo_1' group with `states`
    and `actions`, plus `data` attrs incl. env_info), so downstream tooling reads
    it unchanged. Returns (num_states, was_flagged_successful).
    """
    import robosuite as suite  # 지연 import (버전 태그 기록용, teleop 경로 전용)

    state_paths = os.path.join(ep_dir, "state_*.npz")
    states, actions = [], []
    env_name = None
    success = False
    for state_file in sorted(glob(state_paths)):
        dic = np.load(state_file, allow_pickle=True)
        env_name = str(dic["env"])
        states.extend(dic["states"])
        for ai in dic["action_infos"]:
            actions.append(ai["actions"])
        success = success or bool(dic["successful"])

    if len(states) == 0:
        return 0, success

    # Drop the trailing state: DataCollectionWrapper records the state AFTER the
    # action, so there is one extra state at the end (same fix robosuite applies).
    del states[-1]
    assert len(states) == len(actions), (len(states), len(actions))

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    with h5py.File(hdf5_path, "w") as f:
        grp = f.create_group("data")
        ep_grp = grp.create_group("demo_1")  # one demo per demo_<N>/ folder
        # model.xml is written by DataCollectionWrapper on first interaction, so
        # it should always exist; guard anyway so a missing xml doesn't crash.
        xml_path = os.path.join(ep_dir, "model.xml")
        if os.path.exists(xml_path):
            with open(xml_path, "r") as xml_f:
                ep_grp.attrs["model_file"] = xml_f.read()
        else:
            print(f"[warn] model.xml not found in {ep_dir}; saving without model_file attr.")
        ep_grp.create_dataset("states", data=np.array(states))
        ep_grp.create_dataset("actions", data=np.array(actions))

        now = datetime.datetime.now()
        grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = env_info

    return len(states), success


def next_demo_dir(root):
    """Return the path to the next unused 'demo_<N>' folder under root.

    Scans root for existing 'demo_<int>' folders and returns 'demo_<max+1>'
    (or 'demo_1' if none exist). The folder is NOT created here so that quitting
    without saving leaves no empty stub behind.
    """
    os.makedirs(root, exist_ok=True)
    used = []
    for name in os.listdir(root):
        if name.startswith("demo_") and name[len("demo_"):].isdigit():
            used.append(int(name[len("demo_"):]))
    n = (max(used) + 1) if used else 1
    return os.path.join(root, f"demo_{n}")


def ask_keep_demo():
    """Prompt (Korean) whether to save the just-collected demo.

    Loops until a clear yes/no is given. Returns True to keep+save, False to
    discard and re-collect. Ctrl+D (EOF) is treated as 'no' (re-collect).
    """
    while True:
        try:
            ans = input(">> 이번 데모를 저장하겠습니까? [y/n]: ").strip().lower()
        except EOFError:
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("   y(저장) 또는 n(리셋 후 재수집) 으로 답해주세요.")


def ask_num_demos():
    """Prompt (Korean) for how many demos to collect. Returns a positive int.

    Loops until a positive integer is entered. Ctrl+D (EOF) defaults to 1.
    """
    while True:
        try:
            ans = input(">> 몇 개의 데모를 수집하시겠습니까? (정수 입력): ").strip()
        except EOFError:
            print("   (입력 없음 -> 1개로 진행합니다.)")
            return 1
        if ans.isdigit() and int(ans) > 0:
            return int(ans)
        print("   1 이상의 정수를 입력해주세요.")


# ---------------------------------------------------------------------------
# Version-robust wrapper around robosuite's collect_human_trajectory
# ---------------------------------------------------------------------------
def run_one_episode(env, device, args):
    """Call robosuite's collect_human_trajectory, passing goal_update_mode only
    if this robosuite version's signature accepts it (1.5.2+)."""
    # 지연 import: robosuite 의 teleop 수집 루프(1.5.1 은 goal_update_mode 없음, 1.5.2 는
    # 추가) — inspect.signature 로 버전-강건 호출.
    from robosuite.scripts.collect_human_demonstrations import collect_human_trajectory

    params = inspect.signature(collect_human_trajectory).parameters
    kwargs = {}
    if "goal_update_mode" in params:
        kwargs["goal_update_mode"] = args.goal_update_mode
    collect_human_trajectory(env, device, args.arm, args.max_fr, **kwargs)


# ---------------------------------------------------------------------------
# Collect (interactive teleoperation) — 기존 동작 그대로 (run_collect 로 분리)
# ---------------------------------------------------------------------------
def run_collect(args):
    if args.device != "keyboard":
        raise SystemExit(
            f"--device '{args.device}' is not supported by this utility. Only 'keyboard' is wired here. "
            "Extend it with robosuite's SpaceMouse device if needed."
        )

    # --- interactive (or CLI) environment/robot selection ---
    options = choose_options_interactively(args)

    # --- how many demos to collect (asked in the terminal unless given on CLI) ---
    num_demos = args.num_demos if (args.num_demos and args.num_demos > 0) else ask_num_demos()

    # --- controller config: set the chosen arm controller ---
    # load_composite_controller_config wants a single robot name; use the first.
    first_robot = options["robots"][0] if isinstance(options["robots"], (list, tuple)) else options["robots"]
    controller_config = make_arm_controller_config(args.controller, first_robot)

    # --- build env ---
    env, action_dim, arm_controller_name = build_env(args, options, controller_config)

    # If the user explicitly asked for the 4-dim pipeline action, fail loudly on
    # a mismatch (mirrors collect_pickplace_can.py's hard check).
    if args.controller == "OSC_POSITION" and action_dim != 4:
        env.close()
        raise SystemExit(
            "\n=================== ACTION-SPACE CHECK FAILED ===================\n"
            f"Expected a 4-dim action (OSC_POSITION: 3 pos + 1 gripper) but got "
            f"action_dim = {action_dim} (arm controller: {arm_controller_name}).\n"
            "================================================================="
        )

    # --- output root: ./demos/<Env>_<Robot>/  (each accepted demo -> demo_<N>/) ---
    # e.g. demos/PickPlaceCan_Panda/demo_1/demo.hdf5, demos/PickPlaceMilk_Panda/demo_1/...
    # The Module-1 pipeline's demo_root is ./demos and it discovers these
    # recursively, so demos collected here feed build_feature_bank.py directly.
    robots_tag = "-".join(options["robots"]) if isinstance(options["robots"], (list, tuple)) else options["robots"]
    run_name = "{}_{}".format(options["env_name"], robots_tag)
    root_dir = os.path.join(args.directory, run_name)
    os.makedirs(root_dir, exist_ok=True)

    # --- env metadata for the hdf5 (so downstream code follows THIS demo) ---
    # We also record object_type for PickPlace-family envs (PickPlaceCan -> "can",
    # PickPlaceMilk -> "milk", ...) so build_feature_bank.py picks the right object
    # per demo without any hard-coded "can". Non-PickPlace envs leave it null.
    env_name = options["env_name"]
    object_type = None
    if env_name.startswith("PickPlace") and len(env_name) > len("PickPlace"):
        object_type = env_name[len("PickPlace"):].lower()   # PickPlaceMilk -> "milk"
    env_info_dict = {
        "env_name": env_name,
        "robots": options["robots"] if isinstance(options["robots"], (list, tuple)) else [options["robots"]],
        "controller_configs": controller_config,
        "action_dim": action_dim,
        "arm_controller": arm_controller_name,
        "control_freq": args.control_freq,
    }
    if object_type is not None:
        env_info_dict["object_type"] = object_type
    env_info = json.dumps(env_info_dict)

    # --- wrap: Visualization -> DataCollection (as in the stock collector) ---
    from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper  # 지연 import
    env = VisualizationWrapper(env)
    tmp_directory = os.path.join("/tmp", "rs_collect_{}".format(str(time.time()).replace(".", "_")))
    env = DataCollectionWrapper(env, tmp_directory)

    # --- keyboard device: stock Keyboard for OSC_POSE (rolling), shim for OSC_POSITION ---
    device = _build_teleop_device(args, env)

    print_launch_banner(args, options, root_dir, action_dim, arm_controller_name)

    # --- collection loop: collect exactly `num_demos` accepted demos ---
    # Each iteration runs a single episode (collect_human_trajectory resets the
    # env at its start and calls env.close() when the episode ends -- via 'q' or
    # a 10-step success hold). We then:
    #   * if no data was recorded (quit before moving) -> re-collect silently.
    #   * otherwise ALWAYS prompt to save (independent of the env's auto-success
    #     detection, so the prompt always appears):
    #       'y' -> save this one episode to the NEXT demos/<Env>_<Robot>/demo_<N>/
    #              demo.hdf5 (one demo per folder); count it toward num_demos.
    #       'n' -> discard it; the env resets on the next loop = re-collect.
    # Only accepted ('y') demos count, so a bad take that you reject does not use
    # up one of your num_demos. The loop ends once `saved` reaches num_demos.
    #
    # IMPORTANT: we do NOT delete the episode folders under tmp_directory. On the
    # next env.reset(), DataCollectionWrapper flushes any leftover buffer into the
    # PREVIOUS episode's folder; deleting that folder here would make the next
    # reset write into a missing path -> FileNotFoundError. Each demo_<N>/demo.hdf5
    # still holds exactly one demo because we save one specific ep_dir per accept.
    # tmp_directory lives under /tmp and is best-effort cleaned on exit.
    print(f"[info] 이번 세션에서 {num_demos}개의 데모를 수집합니다.\n")
    saved = 0
    try:
        while saved < num_demos:
            print(f"===== 데모 {saved + 1}/{num_demos} 수집 시작 "
                  f"(창에서 조작, 끝나면 q) =====")
            before = list_episode_dirs(tmp_directory)
            run_one_episode(env, device, args)

            ep_dir = newest_episode_dir(tmp_directory, before)
            if ep_dir is None:
                print("[info] 이번 에피소드에서 기록된 데이터가 없습니다. 다시 수집합니다.\n")
                continue

            # Informational only: whether the env auto-detected task success.
            auto_ok = episode_successful(ep_dir)
            print("[info] 이번 에피소드 task 자동 성공 감지: "
                  f"{'성공(success)' if auto_ok else '미감지(not flagged)'}")

            if ask_keep_demo():
                out_dir = next_demo_dir(root_dir)          # demos/<Env>_<Robot>/demo_<N>
                os.makedirs(out_dir, exist_ok=True)
                n_states, _ = save_episode_as_hdf5(ep_dir, out_dir, env_info)
                if n_states == 0:
                    # No usable transitions -> remove the empty stub folder, retry.
                    shutil.rmtree(out_dir, ignore_errors=True)
                    print("[info] 저장할 스텝이 없어 저장을 건너뜁니다. 다시 수집합니다.\n")
                    continue
                saved += 1
                print(f"\n[saved {saved}/{num_demos}] {os.path.join(out_dir, 'demo.hdf5')}  "
                      f"(states={n_states})\n")
            else:
                print("[info] 저장하지 않고 리셋 후 다시 수집합니다.\n")

        print(f"[done] 목표한 {num_demos}개 데모 수집 완료. 저장 위치: {root_dir}")
    except KeyboardInterrupt:
        print(f"\n[done] 사용자에 의해 중단되었습니다. ({saved}/{num_demos} 저장) "
              f"저장 위치: {root_dir}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        # best-effort cleanup of the raw episode dumps under /tmp
        shutil.rmtree(tmp_directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# SECTION — inspect / extract  (blueprint v5 Sec 4.2–4.5)
# ---------------------------------------------------------------------------
# demo.hdf5 -> raw_demo_XXX.npz (+ manifest). robosuite replay 는 phase_segment(ps) 의
# helper 를 그대로 재사용한다(중복 금지, Sec 1.1). raw npz 스키마는 feature_select.
# read_raw_npz 가 읽는 계약과 정확히 일치해야 하므로 fs.RAW_SCHEMA_VERSION 을 단일 소스로 쓴다.
def _sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _demo_paths(input_dir, pattern="demo.hdf5"):
    paths = sorted(glob(os.path.join(input_dir, "**", pattern), recursive=True))
    if not paths:
        raise SystemExit(f"{input_dir} 아래 '{pattern}' 없음. 먼저 데모를 수집하거나 "
                         f"--input 경로를 확인하세요.")
    return paths


def run_inspect(input_dir, pattern="demo.hdf5"):
    """각 demo.hdf5 의 integrity 검사(Sec 4.2). robosuite env 불필요 — hdf5 만 읽는다.
    states/actions 존재·길이일치·NaN·action-dim·env metadata 확인."""
    paths = _demo_paths(input_dir, pattern)
    print(f"[inspect] {len(paths)} demo file(s) under {input_dir}")
    all_ok, total_eps = True, 0
    for hp in paths:
        try:
            env_info, demos = ps.read_demo(hp)
        except Exception as e:
            all_ok = False
            print(f"  [FAIL] {hp}: read 실패 ({type(e).__name__}: {e})")
            continue
        for name, states, actions, model_xml in demos:
            total_eps += 1
            states = np.asarray(states); actions = np.asarray(actions)
            adim = int(actions.shape[1]) if actions.ndim == 2 else None
            checks = {
                "states": states.size > 0,
                "actions": actions.size > 0,
                "len_match": len(states) == len(actions),
                "no_nan": bool(np.isfinite(states).all() and np.isfinite(actions).all()),
                "action_dim": adim is not None,
                "model_xml": model_xml is not None,
            }
            ok = all(checks.values()); all_ok = all_ok and ok
            fails = [k for k, v in checks.items() if not v]
            print(f"  [{'PASS' if ok else 'FAIL'}] {hp} :: {name}  T={len(states)} "
                  f"adim={adim} env={env_info.get('env_name')} "
                  f"robots={env_info.get('robots')} ctrl={env_info.get('arm_controller')} "
                  f"freq={env_info.get('control_freq')}"
                  + (f"  결함:{fails}" if fails else ""))
    print(f"\n[inspect {'PASS' if all_ok else 'FAIL'}] episodes={total_eps}")
    return all_ok


def resolve_goal_context(env, env_info, obj, tol=0.03):
    """goal 우선순위(Sec 4.3): hdf5 metadata > env target site/bin > 최종물체위치 fallback.
    robosuite 버전/태스크마다 target 접근이 달라 best-effort 로 시도하고, 못 찾으면 obj[-1]
    로 fallback 하되 from_context=False 로 명확히 표시(silent 금지, Sec 15.2).
    ⚠ 실 robosuite 없이 검증 불가 — 실데모에서 맞는 attribute 를 확인해 다듬어야 함."""
    g = (env_info or {}).get("goal") or (env_info or {}).get("target_pos")
    if g is not None and np.ravel(g).size == 3:
        return {"type": "metadata_goal",
                "target_pos": [float(x) for x in np.ravel(g)],
                "position_tolerance": tol}, True
    if env is not None:
        for attr in ("target_bin_placements", "target_pos", "target_bin_pos"):
            try:
                v = getattr(env, attr, None)
                if v is None:
                    continue
                v = np.asarray(v, float)
                if v.ndim == 2 and v.shape[0] >= 1:
                    v = v[0]
                if v.shape == (3,):
                    return {"type": "target_region",
                            "target_pos": [float(x) for x in v],
                            "position_tolerance": tol}, True
            except Exception:
                pass
    return {"type": "final_object_pos_fallback",
            "target_pos": [float(x) for x in np.asarray(obj[-1], float)],
            "position_tolerance": tol}, False


def _state_restore_error(env, states, n_samples=8):
    """저장 state 를 복원 후 다시 flatten 해 max|Δ| 측정(Sec 4.5 state-restore gate).
    robosuite/mujoco 버전에 따라 get_state API 가 달라 best-effort. 측정 불가면 NaN."""
    try:
        idx = np.linspace(0, len(states) - 1, min(n_samples, len(states))).astype(int)
        errs = []
        for i in idx:
            st = np.asarray(states[i], float)
            env.sim.set_state_from_flattened(st); env.sim.forward()
            back = np.asarray(env.sim.get_state().flatten(), float)
            m = min(len(st), len(back))
            errs.append(float(np.max(np.abs(back[:m] - st[:m]))))
        return float(np.max(errs)) if errs else float("nan")
    except Exception as e:
        print(f"    [warn] state-restore 측정 불가: {type(e).__name__}: {e}")
        return float("nan")


def save_raw_npz(output_dir, tag, rep, dt, actions, states, goal_ctx, from_ctx,
                 env_info, src_path, restore_err=float("nan")):
    """replay 결과를 raw_<tag>.npz + raw_<tag>_manifest.json 으로 저장. 스키마는
    feature_select.read_raw_npz 계약과 일치(단일 소스: fs.RAW_SCHEMA_VERSION)."""
    eef, obj, grip, eq, oq, con = rep
    T = len(eef)
    npz_path = os.path.join(output_dir, f"raw_{tag}.npz")
    np.savez(npz_path,
             time=np.arange(T) * dt, actions=np.asarray(actions, float),
             states=np.asarray(states),
             eef_pos=np.asarray(eef, float), eef_quat=np.asarray(eq, float),
             object_pos=np.asarray(obj, float), object_quat=np.asarray(oq, float),
             gripper_aperture=np.asarray(grip, float), contact=np.asarray(con, float),
             goal_context=np.array(goal_ctx, dtype=object),
             goal=np.asarray(goal_ctx["target_pos"], float),
             dt=float(dt), schema_version=fs.RAW_SCHEMA_VERSION)
    acts = np.asarray(actions)
    manifest = {
        "schema_version": fs.RAW_SCHEMA_VERSION,
        "environment_name": env_info.get("env_name"),
        "robot_name": env_info.get("robots"),
        "controller_name": env_info.get("arm_controller"),
        "action_dimension": int(acts.shape[1]) if acts.ndim == 2 else None,
        "control_frequency": env_info.get("control_freq"),
        "object_type": env_info.get("object_type"),
        "source_demo_path": os.path.abspath(src_path),
        "source_demo_checksum": _sha256(src_path),
        "goal_from_context": bool(from_ctx),
        "goal_context": goal_ctx,
        "state_restore_max_error": (None if not np.isfinite(restore_err)
                                    else float(restore_err)),
        "n_steps": int(T),
        "feature_name_order": fs.NAMES,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(output_dir, f"raw_{tag}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return npz_path, manifest


def run_extract(input_dir, output_dir, pattern="demo.hdf5", max_demos=0,
                restore_tol=1e-3):
    """demo.hdf5 -> artifacts/raw/raw_demo_XXX.npz (+manifest). phase_segment 의 replay
    helper 를 재사용한다(robosuite 설치 + 실데모 필요)."""
    paths = _demo_paths(input_dir, pattern)
    if max_demos > 0:
        paths = paths[:max_demos]
    os.makedirs(output_dir, exist_ok=True)
    cache, idx, goal_all, worst = {}, 0, True, 0.0
    try:
        for hp in paths:
            env_info, demos = ps.read_demo(hp)
            robots = env_info["robots"]
            sig = (env_info["env_name"],
                   tuple(robots) if isinstance(robots, (list, tuple)) else (robots,))
            if sig not in cache:
                print(f"[info] env 생성 {sig[0]} / {sig[1]} ...")
                cache[sig] = ps.build_env(env_info)
            env = cache[sig]
            dt = 1.0 / float(env_info.get("control_freq", 20))
            object_type = env_info.get("object_type", None)
            # tag 는 데모 폴더 이름(예: demo_1)으로 -> fcm/degradation 의 demo_by_tag
            # (폴더 이름 키)와 일치해야 downstream 이 skip 하지 않는다. 한 파일에 여러
            # episode 면 ep 접미사를 붙인다.
            folder = os.path.basename(os.path.dirname(hp)) or f"demo_{idx:03d}"
            for ei, (name, states, actions, model_xml) in enumerate(demos):
                ps.reset_to_scene(env, model_xml)
                rep = ps.replay(env, states, object_type)      # (eef,obj,grip,eq,oq,con)
                goal_ctx, from_ctx = resolve_goal_context(env, env_info, rep[1])
                goal_all = goal_all and from_ctx
                rerr = _state_restore_error(env, states)
                if np.isfinite(rerr):
                    worst = max(worst, rerr)
                tag = folder if len(demos) == 1 else f"{folder}_ep{ei}"
                save_raw_npz(output_dir, tag, rep, dt, actions, states,
                             goal_ctx, from_ctx, env_info, hp, restore_err=rerr)
                print(f"  [ok] {hp}::{name} -> raw_{tag}.npz  T={len(rep[0])}  "
                      f"goal_from_context={from_ctx}  restore_err={rerr:.2e}")
                idx += 1
    finally:
        for env in cache.values():
            try:
                env.close()
            except Exception:
                pass
    passed = idx > 0 and goal_all and worst <= restore_tol
    print(f"\n[Stage 1-3 {'PASS' if passed else 'WARN'}]")
    print(f"Extracted raw trajectories: {idx}   -> {output_dir}/raw_demo_*.npz (+manifest)")
    if not goal_all:
        print("[warn] 일부 demo 의 goal 을 env 에서 못 읽어 최종물체위치 fallback 사용"
              "(goal_from_context=False). 실태스크 target 접근을 확인해 resolve_goal_context 를 다듬으세요(Sec 6.6).")
    if worst > restore_tol:
        print(f"[warn] state-restore 최대 오차 {worst:.2e} > tol {restore_tol:.0e} "
              "-> feature/FCM/degradation 신뢰도 저하 가능(Sec 4.5).")
    print(f"Next: python feature_select.py all --input {output_dir} --output artifacts/features")
    return idx > 0


def run_selftest():
    """robosuite 없이 inspect + raw npz 스키마 왕복(save -> fs.read_raw_npz ->
    compute_trajectory)을 검증. 실제 robosuite replay 는 데모+robosuite 환경이 있어야 하며
    여기서는 검증하지 않는다(합성 hdf5 로 hdf5 계약과 npz 스키마만 확인)."""
    import tempfile
    print("=== collect_demo SELFTEST (robosuite 불필요; 합성 hdf5/npz) ===")
    ok = True
    tmp = tempfile.mkdtemp(prefix="collect_selftest_")
    try:
        T, adim, sdim = 40, 7, 45
        states = np.random.default_rng(0).normal(0, 1, (T, sdim))
        actions = np.random.default_rng(1).normal(0, 0.1, (T, adim))
        env_info = {"env_name": "PickPlaceBread", "robots": ["UR5e"],
                    "arm_controller": "OSC_POSE", "control_freq": 20,
                    "object_type": "bread"}
        demo_dir = os.path.join(tmp, "demos", "demo_000")
        os.makedirs(demo_dir, exist_ok=True)
        hp = os.path.join(demo_dir, "demo.hdf5")
        with h5py.File(hp, "w") as f:
            grp = f.create_group("data")
            grp.attrs["env_info"] = json.dumps(env_info)
            grp.attrs["env"] = env_info["env_name"]
            ep = grp.create_group("demo_1")
            ep.attrs["model_file"] = "<mujoco/>"
            ep.create_dataset("states", data=states)
            ep.create_dataset("actions", data=actions)

        if not run_inspect(os.path.join(tmp, "demos")):
            ok = False; print("[FAIL] 합성 demo inspect 가 FAIL")

        rng = np.random.default_rng(2)
        eef = rng.normal(0, 0.1, (T, 3)); obj = rng.normal(0, 0.1, (T, 3))
        grip = (np.arange(T) > T // 2).astype(float)
        eq = np.tile([0.0, 0, 0, 1.0], (T, 1)); oq = np.tile([0.0, 0, 0, 1.0], (T, 1))
        con = grip.copy()
        goal_ctx = {"type": "target_region", "target_pos": [0.5, 0.0, 0.85],
                    "position_tolerance": 0.03}
        out = os.path.join(tmp, "artifacts", "raw"); os.makedirs(out, exist_ok=True)
        npz_path, manifest = save_raw_npz(out, "demo_000",
                                          (eef, obj, grip, eq, oq, con), 0.05,
                                          actions, states, goal_ctx, True, env_info, hp)
        if not os.path.basename(npz_path).startswith("raw_demo_"):
            ok = False; print(f"[FAIL] raw npz 파일명이 feature_select 글롭과 불일치: {npz_path}")

        raw = fs.read_raw_npz(npz_path)
        F = fs.compute_trajectory(raw["eef"], raw["obj"], raw["grip"], raw["dt"],
                                  raw["eef_q"], raw["obj_q"], goal=raw["goal"],
                                  obj0_z=float(raw["obj"][0, 2]),
                                  actions=raw["actions"], contact=raw["contact"])
        if F.shape != (T, fs.N_FEATURES):
            ok = False; print(f"[FAIL] compute_trajectory shape {F.shape} != {(T, fs.N_FEATURES)}")
        if not np.isfinite(F).all():
            ok = False; print("[FAIL] raw->feature 에 NaN/Inf")
        if raw["goal"] is None or not np.allclose(raw["goal"], [0.5, 0.0, 0.85]):
            ok = False; print(f"[FAIL] goal_context 왕복 실패: {raw['goal']}")
        need = {"schema_version", "environment_name", "controller_name",
                "source_demo_checksum", "goal_from_context", "feature_name_order"}
        if not need.issubset(manifest.keys()):
            ok = False; print(f"[FAIL] manifest 키 누락: {need - set(manifest.keys())}")
        print(f"raw npz 스키마 왕복: save -> fs.read_raw_npz -> compute_trajectory {F.shape} ✓")

        gc, from_ctx = resolve_goal_context(None, {}, obj)
        if from_ctx or gc["type"] != "final_object_pos_fallback":
            ok = False; print("[FAIL] goal fallback 이 from_context=False 여야 함")

        bad = os.path.join(out, "raw_bad.npz")
        np.savez(bad, eef_pos=eef)                 # object_pos/gripper_aperture 누락
        try:
            fs.read_raw_npz(bad); ok = False
            print("[FAIL] 필수키 누락인데 read_raw_npz 가 통과함")
        except KeyError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    print("(주의: 실제 robosuite replay 는 데모+robosuite 환경 필요 -> 여기선 미검증. "
          "extract 는 demo.hdf5 가 있을 때 phase_segment.replay 로 동작.)")
    return ok


# ---------------------------------------------------------------------------
# Main — 서브커맨드 dispatch (기본/서브커맨드 없음 = 기존 teleop 수집)
# ---------------------------------------------------------------------------
def _add_collect_args(p):
    p.add_argument("--directory", type=str, default=os.path.join(".", "demos"),
                   help="Root directory to store collected demos.")
    p.add_argument("--environment", type=str, default=None,
                   help="robosuite environment name. Omit to pick from a terminal menu.")
    p.add_argument("--robots", nargs="+", type=str, default=None,
                   help="Robot model(s). Omit to pick from a terminal menu.")
    p.add_argument("--num-demos", type=int, default=None,
                   help="How many demos to collect (saved with 'y'). Omit to be asked.")
    p.add_argument("--env-configuration", type=str, default=None,
                   help="Two-arm configuration (e.g. 'bimanual'). Omit to pick from a menu.")
    p.add_argument("--device", type=str, default="keyboard",
                   help="Teleop device (only 'keyboard' is wired here).")
    p.add_argument("--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"],
                   help="'mjviewer' = free mouse-movable camera; 'mujoco' = OpenCV fixed --camera.")
    p.add_argument("--camera", type=str, default="agentview", help="Initial camera view.")
    p.add_argument("--controller", type=str, default="OSC_POSE",
                   choices=["OSC_POSE", "OSC_POSITION"],
                   help="Arm controller. OSC_POSE=7-dim(roll), OSC_POSITION=4-dim(no roll).")
    p.add_argument("--control-freq", type=int, default=20, help="Env control frequency (Hz).")
    p.add_argument("--max-fr", type=int, default=20, help="Cap collection loop frames/sec.")
    p.add_argument("--pos-sensitivity", type=float, default=1.0, help="Position input sensitivity.")
    p.add_argument("--rot-sensitivity", type=float, default=1.0, help="Rotation input sensitivity.")
    p.add_argument("--arm", type=str, default="right", help="Which arm to control.")
    p.add_argument("--goal-update-mode", type=str, default="target",
                   choices=["target", "achieved"],
                   help="Passed to device.input2action on versions that accept it.")


def main():
    parser = argparse.ArgumentParser(
        description="robosuite demo 수집(teleop) + inspect + raw extract. "
                    "서브커맨드 없이 실행하면 기존 인터랙티브 수집.")
    _add_collect_args(parser)                    # 하위 호환: 옛 플래그로 바로 수집
    sub = parser.add_subparsers(dest="cmd")

    pin = sub.add_parser("inspect", help="demo.hdf5 integrity 검사(robosuite 불필요)")
    pin.add_argument("--input", default=os.path.join(".", "demos"))
    pin.add_argument("--pattern", default="demo.hdf5")

    pex = sub.add_parser("extract",
                         help="demo.hdf5 -> artifacts/raw/raw_demo_XXX.npz (+manifest)")
    pex.add_argument("--input", default=os.path.join(".", "demos"))
    pex.add_argument("--output", default=os.path.join("artifacts", "raw"))
    pex.add_argument("--pattern", default="demo.hdf5")
    pex.add_argument("--max-demos", type=int, default=0)
    pex.add_argument("--restore-tol", type=float, default=1e-3)

    sub.add_parser("collect", help="인터랙티브 teleop 수집(기본과 동일)")
    sub.add_parser("selftest", help="robosuite 없이 스키마/왕복 검증")

    args = parser.parse_args()
    if args.cmd == "inspect":
        run_inspect(args.input, args.pattern)
    elif args.cmd == "extract":
        run_extract(args.input, args.output, args.pattern, args.max_demos, args.restore_tol)
    elif args.cmd == "selftest":
        raise SystemExit(0 if run_selftest() else 1)
    else:                                         # None(bare) 또는 "collect"
        run_collect(args)


if __name__ == "__main__":
    main()