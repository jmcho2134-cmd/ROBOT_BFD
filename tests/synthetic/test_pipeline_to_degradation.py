"""Integration test: the whole pipeline through simulator-validated degradation."""

import numpy as np

from bfd_pipeline.pipeline import run_pipeline


def test_pipeline_produces_monotonic_degradation_families():
    out = run_pipeline(
        n_demos=12, seed=0, verbose=False,
        fcm_ensemble=2, fcm_max_demos=10, screen_demos=4,
    )

    # Gate 8: zero perturbation -> ~zero residual
    assert out["gate8"] < 1e-3

    # segmentation produced a sane, small number of phases
    assert 3 <= out["canonical_K"] <= 8

    # FCM screening actually shrinks the candidate set
    assert out["n_screened"] < out["n_candidates"]

    # at least one simulator-validated degradation family
    families = out["families"]
    assert len(families) >= 1

    for fam in families:
        degs = np.array(fam.degradation_scores)
        # Original level is the zero-degradation anchor
        assert abs(degs[0]) < 1e-2
        # monotone non-decreasing degradation with lambda (within gate tolerance)
        assert fam.monotonicity_score >= 0.75
        # success preserved through the Severe level (all but the final 'Max')
        assert all(fam.task_success[:-1])
        # genuinely graded, not a flat family
        assert degs[-1] - degs[0] > 0.03
