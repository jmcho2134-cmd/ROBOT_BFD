"""Stage 4 - feature engine.

Applies every registered feature function to a RawTrajectory in a fixed order
and produces a FeatureTrajectory. Invalid (non-finite) samples are recorded in
`valid_mask`; the raw value is preserved (set to 0.0 in the array only where the
mask is False, so downstream never multiplies by NaN by accident).
"""

from __future__ import annotations

import numpy as np

from bfd_pipeline.core.types import ArtifactMetadata, FeatureTrajectory, RawTrajectory
from bfd_pipeline.features.normalization import RobustScaler
from bfd_pipeline.features.registry import FeatureRegistry


def compute_feature_trajectory(
    traj: RawTrajectory,
    registry: FeatureRegistry,
    scaler: RobustScaler | None = None,
) -> FeatureTrajectory:
    names = registry.names
    horizon = traj.horizon
    values = np.zeros((horizon, len(names)), dtype=float)
    valid = np.ones((horizon, len(names)), dtype=bool)

    for j, name in enumerate(names):
        fn = registry.function(name)
        signal = np.asarray(fn(traj), dtype=float).reshape(-1)
        if signal.shape[0] != horizon:
            raise ValueError(
                f"feature '{name}' returned length {signal.shape[0]}, "
                f"expected {horizon}"
            )
        finite = np.isfinite(signal)
        valid[:, j] = finite
        signal = np.where(finite, signal, 0.0)
        values[:, j] = signal

    dt = 1.0
    t = np.asarray(traj.time, dtype=float)
    if t.size >= 2:
        d = float(np.median(np.diff(t)))
        dt = d if d > 0 else 1.0

    normalized = scaler.transform(values) if scaler is not None else None

    meta = ArtifactMetadata(
        source_artifact_ids=[traj.episode_id],
        control_frequency=(1.0 / dt) if dt > 0 else None,
        feature_name_order=list(names),
    )
    return FeatureTrajectory(
        episode_id=traj.episode_id,
        raw_values=values,
        names=list(names),
        dt=dt,
        valid_mask=valid,
        normalized_values=normalized,
        metadata=meta,
    )


def fit_scaler_on_trajectories(
    trajectories: list[RawTrajectory],
    registry: FeatureRegistry,
    epsilon: float = 1e-6,
    clip_value: float = 10.0,
) -> RobustScaler:
    """Fit a RobustScaler from a set of TRAIN trajectories."""
    matrices = [
        compute_feature_trajectory(traj, registry, scaler=None).raw_values
        for traj in trajectories
    ]
    return RobustScaler.fit(
        matrices, registry.names, epsilon=epsilon, clip_value=clip_value
    )
