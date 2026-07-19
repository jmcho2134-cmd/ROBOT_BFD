"""Stage 2 gate tests."""

import h5py
import numpy as np
import pytest

from bfd_pipeline.data.demo_io import load_demo_file


def _write_demo(path, n_episodes=2, horizon=30, state_dim=20, action_dim=7,
                inject_nan_in=None):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i in range(n_episodes):
            g = data.create_group(f"demo_{i}")
            states = np.random.randn(horizon, state_dim).astype("float32")
            actions = np.random.randn(horizon, action_dim).astype("float32")
            if inject_nan_in == i:
                states[5, 3] = np.nan
            g.create_dataset("states", data=states)
            g.create_dataset("actions", data=actions)
            g.attrs["success"] = True


def test_demo_loader_shapes(tmp_path):
    path = tmp_path / "demos.hdf5"
    _write_demo(path, n_episodes=3, horizon=40)
    episodes, report = load_demo_file(str(path))
    assert len(episodes) == 3
    assert report.rejected == {}
    for ep in episodes:
        assert ep.states.shape[0] == ep.actions.shape[0] == 40
        assert ep.actions.shape[1] == 7
        ep.validate()  # must not raise


def test_demo_loader_rejects_nan_episode_only(tmp_path):
    path = tmp_path / "demos.hdf5"
    _write_demo(path, n_episodes=3, inject_nan_in=1)
    episodes, report = load_demo_file(str(path))
    assert len(episodes) == 2  # the clean two survive
    assert "demo_1" in report.rejected
    assert "NaN" in report.rejected["demo_1"] or "Inf" in report.rejected["demo_1"]


def test_demo_loader_deterministic(tmp_path):
    path = tmp_path / "demos.hdf5"
    _write_demo(path, n_episodes=2)
    e1, _ = load_demo_file(str(path))
    e2, _ = load_demo_file(str(path))
    for a, b in zip(e1, e2):
        assert np.array_equal(a.states, b.states)
        assert a.episode_id == b.episode_id


def test_demo_loader_empty_file_raises(tmp_path):
    path = tmp_path / "empty.hdf5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(ValueError):
        load_demo_file(str(path))
