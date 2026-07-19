"""Stage 3 - simulator adapter interface.

Concrete robosuite/MuJoCo replay lives in `robosuite_adapter.py` and requires the
`sim` extra plus real demonstration data. This base class fixes the contract that
feature functions and rollout collection depend on, so downstream stages can be
built and tested against it before the heavy simulator is wired in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from bfd_pipeline.core.types import EpisodeData, RawTrajectory


class BaseReplayAdapter(ABC):
    """Restores saved simulator states and extracts semantic raw signals."""

    @abstractmethod
    def action_layout(self) -> dict:
        """Return action subspace indices, e.g.
        {"position": [0,1,2], "rotation": [3,4,5], "gripper": [6]}.
        """

    @abstractmethod
    def replay(self, episode: EpisodeData) -> RawTrajectory:
        """Restore each saved state and extract a RawTrajectory."""

    @abstractmethod
    def restore_error(self, episode: EpisodeData, timestep: int) -> float:
        """Gate 3: max_abs(saved_state - flatten(restore(saved_state)))."""


def synthetic_pick_place_trajectory(
    episode_id: str = "synthetic_0",
    horizon: int = 120,
    dt: float = 0.05,
    seed: int = 0,
) -> RawTrajectory:
    """A physically plausible-ish Pick-and-Place raw trajectory for tests.

    Phases: approach (dist down) -> grasp (contact on) -> lift (height up) ->
    transport (object-goal dist down) -> place (height down). No real physics;
    just structured signals so Stage 4/5/6 have something to chew on.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(horizon) * dt

    p = np.linspace(0.0, 1.0, horizon)  # normalized progress
    target = np.array([0.6, 0.2, 0.05])

    object_pos = np.zeros((horizon, 3))
    eef_pos = np.zeros((horizon, 3))
    contact = np.zeros(horizon)
    aperture = np.ones(horizon)

    # segment boundaries in normalized progress
    b = [0.0, 0.25, 0.4, 0.55, 0.85, 1.0]
    obj_start = np.array([0.2, -0.1, 0.02])

    for i, pi in enumerate(p):
        obj = obj_start.copy()
        eef = obj_start + np.array([0.0, 0.0, 0.15])
        if pi < b[1]:  # approach
            frac = pi / (b[1] - b[0])
            eef = eef + (obj - eef) * frac
        elif pi < b[2]:  # grasp
            eef = obj.copy()
            contact[i] = 1.0
            aperture[i] = 0.2
        elif pi < b[3]:  # lift
            frac = (pi - b[2]) / (b[3] - b[2])
            obj = obj + np.array([0.0, 0.0, 0.15]) * frac
            eef = obj.copy()
            contact[i] = 1.0
            aperture[i] = 0.2
        elif pi < b[4]:  # transport
            frac = (pi - b[3]) / (b[4] - b[3])
            lifted = obj_start + np.array([0.0, 0.0, 0.15])
            obj = lifted + (target + np.array([0.0, 0.0, 0.15]) - lifted) * frac
            eef = obj.copy()
            contact[i] = 1.0
            aperture[i] = 0.2
        else:  # place
            frac = (pi - b[4]) / (b[5] - b[4])
            high = target + np.array([0.0, 0.0, 0.15])
            obj = high + (target - high) * frac
            eef = obj.copy()
            aperture[i] = 1.0
        object_pos[i] = obj
        eef_pos[i] = eef

    noise = 1e-3
    eef_pos += rng.normal(0, noise, eef_pos.shape)
    object_pos += rng.normal(0, noise, object_pos.shape)

    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (horizon, 1))
    actions = np.zeros((horizon, 7))
    actions[:, :3] = np.gradient(eef_pos, dt, axis=0)
    actions[:, 6] = np.where(aperture < 0.5, 1.0, -1.0)

    return RawTrajectory(
        episode_id=episode_id,
        time=t,
        actions=actions,
        eef_pos=eef_pos,
        eef_quat=quat,
        object_pos=object_pos,
        object_quat=quat,
        gripper_aperture=aperture,
        contact=contact,
        goal_context={"type": "target_region", "target_pos": target},
        qvel=rng.normal(0, 0.1, (horizon, 7)),
        success=True,
    )
