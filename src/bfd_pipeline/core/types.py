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

    def signal(self, use_normalized: bool = True) -> np.ndarray:
        if use_normalized and self.normalized_values is not None:
            return self.normalized_values
        return self.raw_values


# --------------------------------------------------------------------------
# Stage 5 - Feature Profiling
# --------------------------------------------------------------------------
@dataclass
class FeatureProfile:
    name: str
    binary_score: float
    event_score: float
    trend_score: float
    plateau_score: float
    changepoint_score: float
    endpoint_consistency: float
    cross_demo_direction_consistency: float
    cross_demo_timing_consistency: float
    noise_robustness: float
    structural_score: float
    structural_uncertainty: float
    confidence: float

    def is_structural(
        self,
        structural_threshold: float,
        confidence_threshold: float,
        robustness_threshold: float,
    ) -> bool:
        return (
            self.structural_score >= structural_threshold
            and self.confidence >= confidence_threshold
            and self.noise_robustness >= robustness_threshold
        )

    def reliability(self) -> float:
        """Weight applied to this feature's boundary evidence in Stage 6.

        Pragmatic v1: how-structural x how-confident. NOTE (blueprint review A):
        this still folds signal strength into the weight; splitting strength
        (carried by the evidence) from reliability (this weight) is deferred.
        """
        return float(
            np.clip(self.structural_score, 0.0, 1.0)
            * np.clip(self.confidence, 0.0, 1.0)
        )


# --------------------------------------------------------------------------
# Stage 6 - Phase Segmentation
# --------------------------------------------------------------------------
@dataclass
class SegmentDescriptor:
    episode_id: str
    local_phase_id: int
    start: int
    end: int
    mean: np.ndarray
    std: np.ndarray
    delta: np.ndarray
    slope: np.ndarray
    endpoint: np.ndarray
    duration_normalized: float


@dataclass
class PhaseSegmentation:
    episode_id: str
    boundaries: list[int]            # [0, b1, ..., T]
    phase_ids: np.ndarray            # (T,)
    boundary_scores: np.ndarray      # (len(boundaries),)
    objective_value: float
    confidence: float = 0.0
    debug_only: bool = False
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    @property
    def num_phases(self) -> int:
        return len(self.boundaries) - 1


@dataclass
class CanonicalPhaseModel:
    canonical_phase_count: int
    canonical_labels: list[str]              # z0, z1, ...
    phase_descriptor_centers: np.ndarray     # (K, D)
    phase_descriptor_scales: np.ndarray      # (K, D)
    demo_alignment_paths: dict               # episode_id -> list[canonical_id per local segment]
    confidence: float = 0.0


# --------------------------------------------------------------------------
# Stage 7 - Subgoal
# --------------------------------------------------------------------------
@dataclass
class PhaseSubgoal:
    canonical_phase_id: int
    change_features: list[dict]
    hold_features: list[dict]
    passive_features: list[str]
    endpoint_center: np.ndarray
    endpoint_scale: np.ndarray
    confidence: float = 0.0


# --------------------------------------------------------------------------
# Stage 8/9 - FCM
# --------------------------------------------------------------------------
@dataclass
class FCMSample:
    episode_id: str
    timestep: int
    canonical_phase_id: int
    phase_progress: float
    feature_t: np.ndarray
    action_t: np.ndarray
    perturbation: np.ndarray
    goal_embedding: np.ndarray
    horizon: int
    residual_target: np.ndarray      # (F,)
    task_success: bool = True


# --------------------------------------------------------------------------
# Stage 10/11/12 - Degradation
# --------------------------------------------------------------------------
@dataclass
class DegradationHypothesis:
    hypothesis_id: str
    canonical_phase_id: int
    target_type: str                 # change_opposition | hold_violation
    target_feature_names: list[str]
    target_definition: dict
    confidence: float = 0.0


@dataclass
class PerturbationCandidate:
    candidate_id: str
    hypothesis_id: str
    canonical_phase_id: int
    start_fraction: float
    duration_fraction: float
    action_subspace: str
    direction: np.ndarray            # unit vector, action_dim
    predicted_degradation: float | None = None
    predicted_uncertainty: float | None = None
    screening_score: float | None = None


@dataclass
class ValidatedDegradationFamily:
    family_id: str
    candidate: PerturbationCandidate
    lambda_max: float
    lambda_levels: list[float]
    degradation_scores: list[float]
    task_success: list[bool]
    monotonicity_score: float
    gradedness_score: float
    recovery_score: float
    confidence: float = 0.0
