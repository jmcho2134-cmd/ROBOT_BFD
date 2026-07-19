"""End-to-end driver: demos -> ... -> validated degradation families (Stages 1-12).

Runs on the synthetic Pick-and-Place simulator so the whole pipeline, including
real perturbation consequences, executes without robosuite/MuJoCo.

    python -m bfd_pipeline.pipeline
    # or: python scripts/run_pipeline_to_degradation.py
"""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from bfd_pipeline.consequence.fcm import evaluate_fcm, train_fcm
from bfd_pipeline.consequence.rollout import build_fcm_dataset
from bfd_pipeline.data.demo_io import load_demo_file
from bfd_pipeline.degradation.candidate import (
    build_candidates,
    make_context,
    screen_candidates,
)
from bfd_pipeline.degradation.hypothesis import generate_hypotheses
from bfd_pipeline.degradation.validator import SimContext, validate_candidates
from bfd_pipeline.envs.synthetic_env import (
    SyntheticReplayAdapter,
    action_layout,
    generate_demo_set,
    write_demo_hdf5,
)
from bfd_pipeline.features.engine import (
    compute_feature_trajectory,
    fit_scaler_on_trajectories,
)
from bfd_pipeline.features.profiler import profile_features
from bfd_pipeline.features.registry import default_registry
from bfd_pipeline.segmentation.segmenter import build_canonical, segment_demo
from bfd_pipeline.segmentation.subgoal_inference import infer_subgoals

WINDOWS = [0.10, 0.20, 0.40]


def run_pipeline(
    n_demos: int = 24,
    seed: int = 0,
    verbose: bool = True,
    fcm_ensemble: int = 5,
    fcm_max_demos: int = 16,
    screen_demos: int = 6,
) -> dict:
    def log(*a):
        if verbose:
            print(*a)

    # Stage 1-2: demos + load
    demos = generate_demo_set(n_demos=n_demos, seed=seed)
    path = os.path.join(tempfile.mkdtemp(), "demos.hdf5")
    write_demo_hdf5(path, demos)
    episodes, report = load_demo_file(path)
    log(f"[1-2] loaded {len(episodes)} demos ({len(report.rejected)} rejected)")

    # Stage 3-4: replay + features
    adapter = SyntheticReplayAdapter()
    reg = default_registry()
    raws = [adapter.replay(e) for e in episodes]
    restore_err = max(adapter.restore_error(e, 0) for e in episodes)
    scaler = fit_scaler_on_trajectories(raws, reg)
    fts = [compute_feature_trajectory(r, reg, scaler=scaler) for r in raws]
    log(f"[3-4] state-restore err={restore_err:.1e}, features {fts[0].raw_values.shape[1]}")

    # Stage 5: profiling
    profs = profile_features(fts, bootstrap=20, seed=seed)
    structural = [n for n, p in profs.items() if p.is_structural(0.55, 0.6, 0.5)]
    log(f"[5] structural features: {structural}")

    # Stage 6: segmentation
    segs = [segment_demo(ft, profs, structural, WINDOWS) for ft in fts]
    canonical = build_canonical(fts, segs, structural)
    counts = [s.num_phases for s in segs]
    log(f"[6] canonical K={canonical.canonical_phase_count} "
        f"(demo phases {min(counts)}-{max(counts)}), conf={canonical.confidence:.2f}")

    # Stage 7: subgoals
    subgoals = infer_subgoals(fts, segs, canonical, profs, structural)
    for k, sg in subgoals.items():
        log(f"    z{k}: change={[c['name'] for c in sg.change_features]} "
            f"hold={[h['name'] for h in sg.hold_features]} conf={sg.confidence:.2f}")

    # Stage 8-9: FCM
    struct_idx = [reg.names.index(n) for n in structural]
    ds = build_fcm_dataset(episodes, fts, segs, canonical, scaler, reg,
                           structural, action_layout(),
                           max_demos=min(fcm_max_demos, n_demos), seed=seed + 1)
    pert_cols = ds.X[:, len(structural) + 7:len(structural) + 14]
    zero = np.all(np.abs(pert_cols) < 1e-9, axis=1)
    gate8 = float(np.abs(ds.Y[zero]).mean()) if zero.any() else float("nan")
    n = len(ds.X)
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    fcm = train_fcm(ds.X[tr], ds.Y[tr], ensemble_size=fcm_ensemble, seed=seed)
    metrics = evaluate_fcm(fcm, ds.X[te], ds.Y[te])
    log(f"[8-9] dataset {ds.X.shape}, Gate8 |resid|@0={gate8:.2e}, "
        f"FCM R2(med)={metrics['r2_median_informative']:.2f} MAE={metrics['mae']:.3f}")

    # Stage 10: hypotheses
    hyps = generate_hypotheses(subgoals)
    log(f"[10] {len(hyps)} hypotheses")

    # Stage 11: candidates + screening
    contexts = [make_context(ft, seg, canonical, struct_idx, ep.actions)
                for ft, seg, ep in zip(fts, segs, episodes)]
    screen_ctx = contexts[:screen_demos]
    candidates = build_candidates(hyps, action_layout(), seed=seed)
    screened = screen_candidates(candidates, hyps, subgoals, fcm, screen_ctx,
                                 structural, top_k_per_hypothesis=4)
    log(f"[11] {len(candidates)} candidates -> {len(screened)} screened")

    # Stage 12: simulator validation
    sim_ctx = [SimContext(ep.episode_id, ep.states, ep.actions,
                          ft.normalized_values[:, struct_idx], seg, canonical, 20.0)
               for ep, ft, seg in zip(episodes, fts, segs)]
    families = validate_candidates(screened, hyps, subgoals, sim_ctx,
                                   reg, scaler, structural)
    log(f"[12] {len(families)} validated degradation families")
    for fam in families[:8]:
        log(f"    {fam.family_id[:28]:28} phase z{fam.candidate.canonical_phase_id} "
            f"{fam.candidate.action_subspace:8} lam_max={fam.lambda_max:.2f} "
            f"mono={fam.monotonicity_score:.2f} grad={fam.gradedness_score:.2f} "
            f"deg={[round(d,2) for d in fam.degradation_scores]}")

    return {
        "structural": structural,
        "canonical_K": canonical.canonical_phase_count,
        "gate8": gate8,
        "fcm_metrics": metrics,
        "n_hypotheses": len(hyps),
        "n_candidates": len(candidates),
        "n_screened": len(screened),
        "families": families,
    }


if __name__ == "__main__":
    run_pipeline()
