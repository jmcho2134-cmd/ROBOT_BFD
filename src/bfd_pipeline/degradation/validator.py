"""Stage 12 - exact simulator validation.

Runs FCM-screened candidates in the (synthetic) simulator, calibrates a
per-candidate lambda_max by bracketing + binary search, builds a nested
Original>Mild>Medium>Severe hierarchy, and keeps only families that stay
successful while degrading monotonically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bfd_pipeline.core.types import (
    DegradationHypothesis,
    PerturbationCandidate,
    PhaseSubgoal,
    ValidatedDegradationFamily,
)
from bfd_pipeline.consequence.rollout import branch_states, phase_end_index
from bfd_pipeline.envs.synthetic_env import SyntheticPickPlaceEnv, raw_from_states
from bfd_pipeline.segmentation.subgoal_inference import degradation_score


@dataclass
class SimContext:
    episode_id: str
    states: np.ndarray
    actions: np.ndarray
    struct_norm: np.ndarray
    seg: object
    canonical: object
    control_frequency: float


def _placement(ctx: SimContext, k: int):
    path = ctx.canonical.demo_alignment_paths.get(ctx.episode_id, [])
    for local_k, cid in enumerate(path):
        if cid == k:
            return ctx.seg.boundaries[local_k], ctx.seg.boundaries[local_k + 1]
    return None


def _rollout(env, ctx, reg, scaler, struct_idx, start_idx, window_len,
             end_idx, direction, lam):
    n_steps = len(ctx.actions) - start_idx
    st = branch_states(env, ctx.states[start_idx], ctx.actions, start_idx,
                       window_len, direction, lam, n_steps)
    used = ctx.actions[start_idx:start_idx + n_steps]
    # clip ratio over the perturbed window
    w = min(window_len, len(used))
    pre = used[:w] + lam * direction
    clip_ratio = float(np.mean(np.abs(pre) > 1.0)) if w > 0 else 0.0
    success = bool(env.is_success(st[-1]))
    raw = raw_from_states("v", st, used, ctx.control_frequency)
    from bfd_pipeline.features.engine import compute_feature_trajectory
    ft = compute_feature_trajectory(raw, reg, scaler=scaler)
    local_end = min(end_idx - start_idx, len(st) - 1)
    pert_end = ft.normalized_values[local_end, struct_idx]
    return pert_end, success, clip_ratio


def validate_candidate(
    candidate: PerturbationCandidate,
    hypothesis: DegradationHypothesis,
    subgoal: PhaseSubgoal,
    ctx: SimContext,
    reg,
    scaler,
    structural_names: list[str],
    *,
    initial_probe: float = 0.15,
    growth: float = 2.0,
    tol: float = 0.02,
    max_lambda: float = 3.0,
    max_clip_ratio: float = 0.25,
    min_effect: float = 0.05,
    min_step_effect: float = 0.02,
    min_monotonicity: float = 0.75,
    level_fracs=(0.0, 0.25, 0.5, 0.75, 1.0),
) -> ValidatedDegradationFamily | None:
    struct_idx = [reg.names.index(n) for n in structural_names]
    env = SyntheticPickPlaceEnv()
    place = _placement(ctx, candidate.canonical_phase_id)
    if place is None:
        return None
    a, b = place
    if b - a < 3:
        return None
    start_idx = min(int(a + candidate.start_fraction * (b - a)), b - 2)
    window_len = max(1, int(candidate.duration_fraction * (b - a)))
    end_idx = min(phase_end_index(ctx.seg, start_idx), len(ctx.states) - 1)
    if end_idx - start_idx < 2:
        return None
    demo_end = ctx.struct_norm[end_idx]

    def deg_success(lam):
        pert_end, success, clip = _rollout(
            env, ctx, reg, scaler, struct_idx, start_idx, window_len,
            end_idx, candidate.direction, lam)
        residual = pert_end - demo_end
        d = degradation_score(subgoal, residual, structural_names,
                              hypothesis.target_feature_names, hypothesis.target_type)
        valid = success and clip <= max_clip_ratio
        return d, success, valid

    # --- bracket lambda_max ---
    lam_valid, lam_invalid = 0.0, None
    lam = initial_probe
    while lam <= max_lambda:
        _, _, valid = deg_success(lam)
        if valid:
            lam_valid = lam
            lam *= growth
        else:
            lam_invalid = lam
            break
    if lam_invalid is None:
        lambda_max = min(lam_valid, max_lambda)
    else:
        lo, hi = lam_valid, lam_invalid
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            _, _, valid = deg_success(mid)
            if valid:
                lo = mid
            else:
                hi = mid
        lambda_max = lo
    if lambda_max <= 1e-6:
        return None

    # effect threshold at lambda_max
    d_max, _, _ = deg_success(lambda_max)
    if d_max < min_effect:
        return None

    # --- nested levels ---
    levels = [f * lambda_max for f in level_fracs]
    degs, succ = [], []
    for lam in levels:
        d, s, _ = deg_success(lam)
        degs.append(d)
        succ.append(s)

    # monotonicity (Spearman-like ordered fraction over lambda vs degradation)
    pairs = [(degs[i] <= degs[j]) for i in range(len(degs)) for j in range(i + 1, len(degs))]
    monotonicity = float(np.mean(pairs)) if pairs else 0.0
    # gradedness: at least two consecutive steps exceed min_step_effect
    steps = np.diff(degs)
    gradedness = float(np.mean(steps > min_step_effect))
    # success preservation up to Severe (exclude final 'Max' level)
    severe_ok = all(succ[:-1])

    if monotonicity < min_monotonicity or not severe_ok or (steps > min_step_effect).sum() < 2:
        return None

    recovery = float(np.mean(succ))
    confidence = float(monotonicity * (0.5 + 0.5 * gradedness))
    return ValidatedDegradationFamily(
        family_id=f"{candidate.candidate_id}@{ctx.episode_id}",
        candidate=candidate,
        lambda_max=float(lambda_max),
        lambda_levels=[float(x) for x in levels],
        degradation_scores=[float(x) for x in degs],
        task_success=[bool(x) for x in succ],
        monotonicity_score=monotonicity,
        gradedness_score=gradedness,
        recovery_score=recovery,
        confidence=confidence,
    )


def validate_candidates(
    candidates, hypotheses, subgoals, sim_contexts, reg, scaler,
    structural_names, **kw,
) -> list[ValidatedDegradationFamily]:
    hyp_by_id = {h.hypothesis_id: h for h in hypotheses}
    families: list[ValidatedDegradationFamily] = []
    for c in candidates:
        h = hyp_by_id[c.hypothesis_id]
        sg = subgoals[c.canonical_phase_id]
        # validate on the first demo that contains this phase
        for ctx in sim_contexts:
            if _placement(ctx, c.canonical_phase_id) is None:
                continue
            fam = validate_candidate(c, h, sg, ctx, reg, scaler,
                                     structural_names, **kw)
            if fam is not None:
                families.append(fam)
            break
    return families
