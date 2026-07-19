"""Stage 4 / Gate B tests: feature-function-only registry."""

import numpy as np
import pytest

from bfd_pipeline.features.registry import FeatureRegistry, default_registry


def test_default_registry_order_is_fixed():
    reg = default_registry()
    expected = [
        "eef_object_dist", "object_goal_dist", "object_height",
        "gripper_aperture", "contact", "eef_speed", "eef_jerk",
        "joint_energy", "path_increment", "object_slip",
    ]
    assert reg.names == expected


def test_registry_rejects_semantic_names():
    reg = FeatureRegistry()
    for bad in ("progress_x", "is_event", "boundary_score", "higher_is_worse"):
        with pytest.raises(ValueError):
            reg.register(bad, lambda t: np.zeros(1))


def test_registry_rejects_duplicate():
    reg = FeatureRegistry()
    reg.register("foo", lambda t: np.zeros(1))
    with pytest.raises(ValueError):
        reg.register("foo", lambda t: np.zeros(1))


def test_registry_rejects_non_callable():
    reg = FeatureRegistry()
    with pytest.raises(ValueError):
        reg.register("foo", 123)  # type: ignore[arg-type]
