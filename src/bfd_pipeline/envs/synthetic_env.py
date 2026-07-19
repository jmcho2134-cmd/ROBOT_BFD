"""A lightweight deterministic Pick-and-Place simulator.

This is NOT robosuite. It exists so the whole pipeline through degradation
(Stages 8-12) can run and be validated *with genuine action consequences*
before MuJoCo + real demos are available. It implements the same
`BaseReplayAdapter` contract, so swapping in a real robosuite adapter later
does not touch Stages 5-12.

State layout (12,):  [eef(3), object(3), grasped(1), gripper(1), target(3)]
Action layout (7,):  [dx, dy, dz, droll, dpitch, dyaw, gripper]  (OSC_POSE-like)
Rotation dims [3:6] are inert here (documented limitation).
"""

from __future__ import annotations

import numpy as np

from bfd_pipeline.core.types import ArtifactMetadata, EpisodeData, RawTrajectory
from bfd_pipeline.envs.base_adapter import BaseReplayAdapter

POS_GAIN = 0.03
GRASP_RADIUS = 0.04
TABLE_Z = 0.02
FALL_GAIN = 0.03
WORKSPACE_LOW = np.array([-0.5, -0.5, 0.0])
WORKSPACE_HIGH = np.array([0.9, 0.9, 0.6])
SUCCESS_TOL = 0.06
ACTION_DIM = 7
STATE_DIM = 12


def action_layout() -> dict:
    return {"position": [0, 1, 2], "rotation": [3, 4, 5], "gripper": [6]}


# ----------------------------- state helpers ------------------------------
def make_state(eef, obj, grasped, gripper, target) -> np.ndarray:
    s = np.zeros(STATE_DIM, dtype=float)
    s[0:3] = eef
    s[3:6] = obj
    s[6] = 1.0 if grasped else 0.0
    s[7] = gripper
    s[9:12] = target
    return s


def decode_state(s: np.ndarray) -> dict:
    return {
        "eef": s[0:3].copy(),
        "object": s[3:6].copy(),
        "grasped": bool(s[6] > 0.5),
        "gripper": float(s[7]),
        "target": s[9:12].copy(),
    }


class SyntheticPickPlaceEnv:
    def __init__(self) -> None:
        self.s = np.zeros(STATE_DIM, dtype=float)

    def set_state(self, s: np.ndarray) -> None:
        self.s = np.asarray(s, dtype=float).copy()

    def get_state(self) -> np.ndarray:
        return self.s.copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        d = decode_state(self.s)
        eef = d["eef"] + a[0:3] * POS_GAIN
        eef = np.clip(eef, WORKSPACE_LOW, WORKSPACE_HIGH)

        obj = d["object"]
        grasped = d["grasped"]
        closing = a[6] > 0.0
        if closing:
            gripper = 0.2
            if not grasped and np.linalg.norm(eef - obj) < GRASP_RADIUS:
                grasped = True
        else:
            gripper = 1.0
            grasped = False

        if grasped:
            obj = eef.copy()
        else:
            # gravity: unheld object settles to the table
            obj = obj.copy()
            obj[2] = max(obj[2] - FALL_GAIN, TABLE_Z)

        self.s = make_state(eef, obj, grasped, gripper, d["target"])
        return self.s

    def is_success(self, s: np.ndarray | None = None) -> bool:
        d = decode_state(self.s if s is None else s)
        placed = np.linalg.norm(d["object"] - d["target"]) < SUCCESS_TOL
        released = not d["grasped"]
        on_table = d["object"][2] <= TABLE_Z + 0.02
        return bool(placed and released and on_table)

    def rollout(self, start_state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Run an action sequence from a start state; return (T+1, STATE_DIM) states."""
        self.set_state(start_state)
        out = [self.get_state()]
        for a in actions:
            out.append(self.step(a))
        return np.asarray(out)


# ----------------------------- scripted expert ----------------------------
def _waypoints(obj0: np.ndarray, target: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """(position, gripper_cmd) waypoints. gripper_cmd: -1 open, +1 close."""
    up = np.array([0.0, 0.0, 0.18])
    return [
        (obj0 + up, -1),          # above object, open
        (obj0, -1),               # descend to object, open
        (obj0, +1),               # close -> grasp
        (obj0 + up, +1),          # lift
        (target + up, +1),        # transport above target
        (target, +1),             # descend to target
        (target, -1),             # release
        (target + up * 0.5, -1),  # retreat
    ]


def scripted_demo(
    obj0: np.ndarray,
    target: np.ndarray,
    max_steps: int = 200,
    suboptimal: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Generate one demonstration. Returns (states (T+1,), actions (T,), success)."""
    rng = np.random.default_rng(seed)
    env = SyntheticPickPlaceEnv()
    eef0 = obj0 + np.array([0.05, -0.05, 0.25])
    env.set_state(make_state(eef0, obj0, False, 1.0, target))

    wps = _waypoints(obj0, target)
    wp_idx = 0
    states = [env.get_state()]
    actions = []

    # suboptimality knobs
    pause_left = 0
    detour = np.zeros(3)
    detour_left = 0

    for _ in range(max_steps):
        d = decode_state(env.get_state())
        wp, grip = wps[wp_idx]

        if suboptimal and pause_left == 0 and detour_left == 0 and rng.random() < 0.04:
            if rng.random() < 0.5:
                pause_left = rng.integers(2, 6)
            else:
                detour = rng.normal(0, 0.06, 3)
                detour[2] = abs(detour[2])
                detour_left = rng.integers(3, 8)

        goal = wp.copy()
        if detour_left > 0:
            goal = goal + detour
            detour_left -= 1

        err = goal - d["eef"]
        if pause_left > 0:
            pos_cmd = np.zeros(3)
            pause_left -= 1
        else:
            pos_cmd = np.clip(err / POS_GAIN, -1.0, 1.0)
            if suboptimal:
                pos_cmd = np.clip(pos_cmd + rng.normal(0, 0.15, 3), -1.0, 1.0)

        action = np.zeros(ACTION_DIM)
        action[0:3] = pos_cmd
        action[6] = float(grip)
        env.step(action)
        states.append(env.get_state())
        actions.append(action)

        # advance waypoint when close (and grasp/release settled)
        reached = np.linalg.norm(err) < 0.025 and pause_left == 0 and detour_left == 0
        if reached:
            wp_idx += 1
            if wp_idx >= len(wps):
                break

    states = np.asarray(states)
    actions = np.asarray(actions)
    success = env.is_success(states[-1])
    return states, actions, success


def generate_demo_set(
    n_demos: int = 24,
    seed: int = 0,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Return list of (episode_id, states, actions, target) for successful demos."""
    rng = np.random.default_rng(seed)
    demos = []
    attempts = 0
    while len(demos) < n_demos and attempts < n_demos * 6:
        attempts += 1
        obj0 = np.array([
            rng.uniform(0.05, 0.30),
            rng.uniform(-0.20, 0.10),
            TABLE_Z,
        ])
        target = np.array([
            rng.uniform(0.45, 0.75),
            rng.uniform(0.05, 0.30),
            TABLE_Z,
        ])
        states, actions, success = scripted_demo(
            obj0, target, suboptimal=True, seed=int(rng.integers(0, 1 << 31))
        )
        if success:
            demos.append((f"demo_{len(demos)}", states[:-1], actions, target))
    return demos


def write_demo_hdf5(path: str, demos, control_frequency: float = 20.0) -> None:
    import h5py

    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["control_frequency"] = control_frequency
        for episode_id, states, actions, target in demos:
            g = data.create_group(episode_id)
            g.create_dataset("states", data=states.astype("float32"))
            g.create_dataset("actions", data=actions.astype("float32"))
            g.attrs["success"] = True
            g.attrs["target"] = target.astype("float32")


# ----------------------------- replay adapter -----------------------------
class SyntheticReplayAdapter(BaseReplayAdapter):
    """Decodes saved synthetic states into RawTrajectory (Stage 3 for the toy env)."""

    def __init__(self, control_frequency: float = 20.0) -> None:
        self.control_frequency = control_frequency

    def action_layout(self) -> dict:
        return action_layout()

    def restore_error(self, episode: EpisodeData, timestep: int) -> float:
        # states ARE the exact simulator state; restore is identity.
        return 0.0

    def replay(self, episode: EpisodeData) -> RawTrajectory:
        return raw_from_states(
            episode.episode_id, episode.states, episode.actions,
            self.control_frequency, success=bool(episode.success),
        )


def raw_from_states(
    episode_id: str,
    states: np.ndarray,
    actions: np.ndarray,
    control_frequency: float,
    success: bool = True,
) -> RawTrajectory:
    """Decode a (T, STATE_DIM) state sequence into a RawTrajectory.

    Shared by the replay adapter (Stage 3) and by branch rollouts (Stage 8/12).
    """
    horizon = len(states)
    dt = 1.0 / control_frequency
    time = np.arange(horizon) * dt
    eef = np.stack([decode_state(s)["eef"] for s in states])
    obj = np.stack([decode_state(s)["object"] for s in states])
    grasped = np.array([decode_state(s)["grasped"] for s in states], dtype=float)
    gripper = np.array([decode_state(s)["gripper"] for s in states], dtype=float)
    target = decode_state(states[0])["target"]
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (horizon, 1))
    if len(actions) < horizon:  # pad to length for feature funcs that read actions
        pad = np.zeros((horizon - len(actions), actions.shape[1] if actions.ndim == 2 else ACTION_DIM))
        actions = np.concatenate([actions, pad]) if len(actions) else pad
    return RawTrajectory(
        episode_id=episode_id,
        time=time,
        actions=actions[:horizon],
        eef_pos=eef,
        eef_quat=quat,
        object_pos=obj,
        object_quat=quat,
        gripper_aperture=gripper,
        contact=grasped,
        goal_context={"type": "target_region", "target_pos": target,
                      "position_tolerance": SUCCESS_TOL},
        simulator_states=states,
        success=success,
        metadata=ArtifactMetadata(
            control_frequency=control_frequency,
            source_artifact_ids=[episode_id],
        ),
    )
