"""Common typed data contracts shared across pipeline stages.

Only the structures needed by the currently implemented stages (1-4) live here
in concrete form. Downstream structures (FeatureProfile, PhaseSegmentation, ...)
are added as their stages are implemented, per docs/BfD_code_pipeline_overview.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCHEMA_VERSION = "0.1.0"


@dataclass
class ArtifactMetadata:
    """Provenance attached to every artifact for reproducibility."""

    schema_version: str = SCHEMA_VERSION
    created_at: str | None = None
    git_commit: str | None = None
    random_seed: int | None = None
    config_hash: str | None = None
    source_artifact_ids: list[str] = field(default_factory=list)
    environment_name: str | None = None
    robot_name: str | None = None
    controller_name: str | None = None
    control_frequency: float | None = None
    feature_name_order: list[str] = field(default_factory=list)


@dataclass
class EpisodeData:
    """A single loaded demonstration episode (Stage 2 output)."""

    episode_id: str
    states: np.ndarray            # (T, state_dim)
    actions: np.ndarray           # (T, action_dim)
    model_xml: str | None
    env_info: dict
    success: bool
    source_path: str
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    def validate(self) -> None:
        """Stage 2 core invariants. Raises ValueError on violation."""
        if self.states.ndim != 2:
            raise ValueError(f"states must be 2D, got shape {self.states.shape}")
        if self.actions.ndim != 2:
            raise ValueError(f"actions must be 2D, got shape {self.actions.shape}")
        if len(self.states) != len(self.actions):
            raise ValueError(
                f"length mismatch: states={len(self.states)} actions={len(self.actions)}"
            )
        if len(self.states) == 0:
            raise ValueError("empty episode")
        if not np.isfinite(self.states).all():
            raise ValueError("states contain NaN/Inf")
        if not np.isfinite(self.actions).all():
            raise ValueError("actions contain NaN/Inf")


@dataclass
class RawTrajectory:
    """Semantic signals extracted from a restored simulator rollout (Stage 3 output).

    Feature functions (Stage 4) read from this, never from raw simulator internals.
    """

    episode_id: str
    time: np.ndarray
    actions: np.ndarray

    eef_pos: np.ndarray           # (T, 3)
    eef_quat: np.ndarray          # (T, 4) xyzw
    object_pos: np.ndarray        # (T, 3)
    object_quat: np.ndarray       # (T, 4) xyzw
    gripper_aperture: np.ndarray  # (T,)
    contact: np.ndarray           # (T,) in {0,1} or continuous force proxy

    goal_context: dict = field(default_factory=dict)
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    simulator_states: np.ndarray | None = None
    actuator_proxy: np.ndarray | None = None
    success: bool = True
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    @property
    def horizon(self) -> int:
        return int(len(self.time))


@dataclass
class FeatureTrajectory:
    """Per-timestep feature values in a fixed name order (Stage 4 output)."""

    episode_id: str
    raw_values: np.ndarray            # (T, F)
    names: list[str]                  # length F
    dt: float
    valid_mask: np.ndarray            # (T, F) bool
    normalized_values: np.ndarray | None = None  # (T, F), filled by Stage 4 scaler
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    def __post_init__(self) -> None:
        t, f = self.raw_values.shape
        if len(self.names) != f:
            raise ValueError(
                f"names length {len(self.names)} != feature dim {f}"
            )
        if self.valid_mask.shape != self.raw_values.shape:
            raise ValueError("valid_mask shape must match raw_values shape")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate feature names: {self.names}")
