"""Stage 4 - feature computation functions.

Each function is the ONLY thing a human writes for a feature. It maps a
`RawTrajectory` to a length-T signal. Functions must be pure and must NOT
encode any role/direction semantics (no `higher_is_worse`, no phase labels).

Convention: a function returns a float array of shape (T,). Non-finite entries
are allowed; the engine records them in `valid_mask` rather than silently
dropping them.
"""

from __future__ import annotations

import numpy as np

from bfd_pipeline.core.types import RawTrajectory


def _dt(traj: RawTrajectory) -> float:
    t = np.asarray(traj.time, dtype=float)
    if t.size < 2:
        return 1.0
    d = float(np.median(np.diff(t)))
    return d if d > 0 else 1.0


def _diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Time derivative with edge padding so output length == input length."""
    d = np.gradient(x, dt, axis=0)
    return d


def eef_object_dist(traj: RawTrajectory) -> np.ndarray:
    return np.linalg.norm(traj.eef_pos - traj.object_pos, axis=1)


def object_goal_dist(traj: RawTrajectory) -> np.ndarray:
    target = traj.goal_context.get("target_pos")
    if target is None:
        return np.full(traj.horizon, np.nan)
    target = np.asarray(target, dtype=float).reshape(1, 3)
    return np.linalg.norm(traj.object_pos - target, axis=1)


def object_height(traj: RawTrajectory) -> np.ndarray:
    return np.asarray(traj.object_pos[:, 2], dtype=float)


def gripper_aperture(traj: RawTrajectory) -> np.ndarray:
    return np.asarray(traj.gripper_aperture, dtype=float)


def contact(traj: RawTrajectory) -> np.ndarray:
    return np.asarray(traj.contact, dtype=float)


def eef_speed(traj: RawTrajectory) -> np.ndarray:
    dt = _dt(traj)
    vel = _diff(np.asarray(traj.eef_pos, dtype=float), dt)
    return np.linalg.norm(vel, axis=1)


def eef_jerk(traj: RawTrajectory) -> np.ndarray:
    dt = _dt(traj)
    pos = np.asarray(traj.eef_pos, dtype=float)
    vel = _diff(pos, dt)
    acc = _diff(vel, dt)
    jerk = _diff(acc, dt)
    return np.linalg.norm(jerk, axis=1)


def joint_energy(traj: RawTrajectory) -> np.ndarray:
    """Kinetic-energy proxy: sum of squared joint velocities.

    Falls back to squared action magnitude when qvel is unavailable so the
    feature is still defined (marked valid) rather than all-NaN.
    """
    if traj.qvel is not None:
        qvel = np.asarray(traj.qvel, dtype=float)
        return np.sum(qvel * qvel, axis=1)
    act = np.asarray(traj.actions, dtype=float)
    return np.sum(act * act, axis=1)


def path_increment(traj: RawTrajectory) -> np.ndarray:
    """Per-step end-effector path length (0 at t=0)."""
    pos = np.asarray(traj.eef_pos, dtype=float)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return np.concatenate([[0.0], step])


def object_slip(traj: RawTrajectory) -> np.ndarray:
    """Relative end-effector/object motion per step, gated by contact.

    While grasped, a well-behaved trajectory moves eef and object together;
    slip is the residual relative displacement.
    """
    eef = np.asarray(traj.eef_pos, dtype=float)
    obj = np.asarray(traj.object_pos, dtype=float)
    rel = np.diff(obj, axis=0) - np.diff(eef, axis=0)
    slip = np.linalg.norm(rel, axis=1)
    slip = np.concatenate([[0.0], slip])
    c = np.asarray(traj.contact, dtype=float)
    return slip * c


# Default registry contents. Roles/directions are intentionally absent.
DEFAULT_FEATURE_FUNCTIONS = {
    "eef_object_dist": eef_object_dist,
    "object_goal_dist": object_goal_dist,
    "object_height": object_height,
    "gripper_aperture": gripper_aperture,
    "contact": contact,
    "eef_speed": eef_speed,
    "eef_jerk": eef_jerk,
    "joint_energy": joint_energy,
    "path_increment": path_increment,
    "object_slip": object_slip,
}
