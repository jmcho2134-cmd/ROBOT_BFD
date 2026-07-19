"""Stage 4 feature engine tests on a synthetic Pick-and-Place trajectory."""

import numpy as np

from bfd_pipeline.envs.base_adapter import synthetic_pick_place_trajectory
from bfd_pipeline.features.engine import (
    compute_feature_trajectory,
    fit_scaler_on_trajectories,
)
from bfd_pipeline.features.registry import default_registry


def test_feature_trajectory_shape_and_order():
    traj = synthetic_pick_place_trajectory(horizon=100)
    reg = default_registry()
    ft = compute_feature_trajectory(traj, reg)
    assert ft.raw_values.shape == (100, len(reg))
    assert ft.names == reg.names
    assert ft.valid_mask.shape == ft.raw_values.shape


def test_object_goal_dist_valid_when_target_present():
    traj = synthetic_pick_place_trajectory()
    reg = default_registry()
    ft = compute_feature_trajectory(traj, reg)
    j = ft.names.index("object_goal_dist")
    assert ft.valid_mask[:, j].all()
    # object ends near the target -> final distance small
    assert ft.raw_values[-1, j] < ft.raw_values[0, j]


def test_object_goal_dist_invalid_without_target():
    traj = synthetic_pick_place_trajectory()
    traj.goal_context = {}  # no target_pos
    reg = default_registry()
    ft = compute_feature_trajectory(traj, reg)
    j = ft.names.index("object_goal_dist")
    assert not ft.valid_mask[:, j].any()


def test_eef_object_dist_decreases_on_approach():
    traj = synthetic_pick_place_trajectory()
    reg = default_registry()
    ft = compute_feature_trajectory(traj, reg)
    j = ft.names.index("eef_object_dist")
    # during grasp the eef reaches the object -> near zero mid-trajectory
    assert ft.raw_values[:, j].min() < 0.05


def test_contact_is_binary_like():
    traj = synthetic_pick_place_trajectory()
    reg = default_registry()
    ft = compute_feature_trajectory(traj, reg)
    j = ft.names.index("contact")
    vals = np.unique(ft.raw_values[:, j])
    assert set(np.round(vals, 6)).issubset({0.0, 1.0})


def test_scaler_no_leakage_and_clip():
    train = [synthetic_pick_place_trajectory(seed=s) for s in range(4)]
    test = synthetic_pick_place_trajectory(seed=99)
    reg = default_registry()
    scaler = fit_scaler_on_trajectories(train, reg)
    ft = compute_feature_trajectory(test, reg, scaler=scaler)
    assert ft.normalized_values is not None
    assert np.isfinite(ft.normalized_values).all()
    assert ft.normalized_values.max() <= scaler.clip_value + 1e-9
    assert ft.normalized_values.min() >= -scaler.clip_value - 1e-9
