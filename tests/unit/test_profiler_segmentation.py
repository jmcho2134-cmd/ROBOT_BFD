"""Stage 5-7 tests on synthetic demos (profiler, segmentation, subgoals)."""

import numpy as np
import pytest

from bfd_pipeline.envs.synthetic_env import (
    SyntheticReplayAdapter,
    generate_demo_set,
    write_demo_hdf5,
)
from bfd_pipeline.data.demo_io import load_demo_file
from bfd_pipeline.features.engine import (
    compute_feature_trajectory,
    fit_scaler_on_trajectories,
)
from bfd_pipeline.features.profiler import profile_features
from bfd_pipeline.features.registry import default_registry
from bfd_pipeline.segmentation.segmenter import build_canonical, segment_demo
from bfd_pipeline.segmentation.subgoal_inference import (
    degradation_score,
    infer_subgoals,
)

WINDOWS = [0.10, 0.20, 0.40]


@pytest.fixture(scope="module")
def pipeline_state(tmp_path_factory):
    demos = generate_demo_set(n_demos=16, seed=3)
    path = tmp_path_factory.mktemp("d") / "demos.hdf5"
    write_demo_hdf5(str(path), demos)
    episodes, _ = load_demo_file(str(path))
    adapter = SyntheticReplayAdapter()
    reg = default_registry()
    raws = [adapter.replay(e) for e in episodes]
    scaler = fit_scaler_on_trajectories(raws, reg)
    fts = [compute_feature_trajectory(r, reg, scaler=scaler) for r in raws]
    profs = profile_features(fts, bootstrap=15, seed=1)
    structural = [n for n, p in profs.items() if p.is_structural(0.55, 0.6, 0.5)]
    segs = [segment_demo(ft, profs, structural, WINDOWS) for ft in fts]
    canonical = build_canonical(fts, segs, structural)
    subgoals = infer_subgoals(fts, segs, canonical, profs, structural)
    return dict(fts=fts, profs=profs, structural=structural, segs=segs,
                canonical=canonical, subgoals=subgoals)


def test_structural_features_include_task_markers(pipeline_state):
    s = pipeline_state["structural"]
    # the two distance features are the clearest phase markers
    assert "eef_object_dist" in s
    assert "object_goal_dist" in s
    # jerk (pure quality/noise) must not be treated as structural
    assert "eef_jerk" not in s


def test_segmentation_respects_min_phase_and_bounds(pipeline_state):
    for seg in pipeline_state["segs"]:
        assert seg.boundaries[0] == 0
        assert seg.boundaries[-1] == len(seg.phase_ids)
        assert seg.boundaries == sorted(seg.boundaries)
        lengths = np.diff(seg.boundaries)
        assert lengths.min() >= 4  # ~min_phase_sec/dt


def test_canonical_phase_count_reasonable_and_stable(pipeline_state):
    can = pipeline_state["canonical"]
    assert 3 <= can.canonical_phase_count <= 8
    assert can.confidence >= 0.7  # most demos within +/-1 of K


def test_every_phase_has_change_or_hold(pipeline_state):
    for k, sg in pipeline_state["subgoals"].items():
        assert len(sg.change_features) + len(sg.hold_features) >= 1


def test_degradation_score_sign(pipeline_state):
    sg = next(iter(pipeline_state["subgoals"].values()))
    if not sg.change_features:
        pytest.skip("no change feature in first phase")
    names = pipeline_state["structural"]
    cf = sg.change_features[0]
    j = names.index(cf["name"])
    resid = np.zeros(len(names))
    # residual opposing the demonstrated direction -> positive degradation
    resid[j] = -cf["demo_direction"]
    d = degradation_score(sg, resid, names, [cf["name"]], "change_opposition")
    assert d > 0
