"""Stage 11 - candidate generation and FCM screening.

Each hypothesis spawns action-space perturbation candidates (subspace x direction
x start/duration). The FCM predicts each candidate's structural residual in a few
representative demo contexts; candidates are ranked by predicted degradation minus
an uncertainty penalty, then diversified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bfd_pipeline.core.types import (
    DegradationHypothesis,
    PerturbationCandidate,
    PhaseSubgoal,
)
from bfd_pipeline.consequence.rollout import canonical_phase_ids, phase_end_index
from bfd_pipeline.segmentation.subgoal_inference import degradation_score


@dataclass
class PhaseContext:
    """Per-demo tensors needed to place a candidate and query the FCM."""
    episode_id: str
    struct_norm: np.ndarray      # (T, F_struct)
    actions: np.ndarray          # (T, 7)
    seg: object
    canonical: object
    cids: np.ndarray             # (T,) canonical id per timestep


def make_context(ft, seg, canonical, struct_idx, actions) -> PhaseContext:
    return PhaseContext(
        episode_id=ft.episode_id,
        struct_norm=ft.normalized_values[:, struct_idx],
        actions=actions,
        seg=seg,
        canonical=canonical,
        cids=canonical_phase_ids(seg, canonical, ft.episode_id),
    )


def _local_segment_for_phase(ctx: PhaseContext, k: int):
    """Return (a, b) of the first local segment aligned to canonical phase k."""
    path = ctx.canonical.demo_alignment_paths.get(ctx.episode_id, [])
    for local_k, cid in enumerate(path):
        if cid == k:
            return ctx.seg.boundaries[local_k], ctx.seg.boundaries[local_k + 1]
    return None


def build_candidates(
    hypotheses: list[DegradationHypothesis],
    layout: dict,
    start_fractions=(0.0, 0.5),
    duration_fractions=(0.5, 1.0),
    n_random_dirs: int = 2,
    seed: int = 0,
) -> list[PerturbationCandidate]:
    rng = np.random.default_rng(seed)
    subspaces = {
        "position": layout["position"],
        "gripper": layout["gripper"],
    }
    cands: list[PerturbationCandidate] = []
    cid_counter = 0
    for h in hypotheses:
        dirs = []
        for sname, idxs in subspaces.items():
            for i in idxs:
                for sign in (+1.0, -1.0):
                    d = np.zeros(7)
                    d[i] = sign
                    dirs.append((sname, d))
            for _ in range(n_random_dirs if sname == "position" else 0):
                d = np.zeros(7)
                v = rng.normal(0, 1, len(idxs))
                d[idxs] = v / (np.linalg.norm(v) + 1e-9)
                dirs.append((sname, d))
        for sname, d in dirs:
            for sf in start_fractions:
                for df in duration_fractions:
                    cands.append(PerturbationCandidate(
                        candidate_id=f"cand_{cid_counter}",
                        hypothesis_id=h.hypothesis_id,
                        canonical_phase_id=h.canonical_phase_id,
                        start_fraction=sf,
                        duration_fraction=df,
                        action_subspace=sname,
                        direction=d,
                    ))
                    cid_counter += 1
    return cands


def _fcm_row(feat_t, act_t, perturbation, cid, progress, horizon):
    return np.concatenate([feat_t, act_t, perturbation, [cid], [progress], [horizon]])


def screen_candidates(
    candidates: list[PerturbationCandidate],
    hypotheses: list[DegradationHypothesis],
    subgoals: dict[int, PhaseSubgoal],
    fcm,
    contexts: list[PhaseContext],
    structural_names: list[str],
    lambda_probe: float = 0.15,
    uncertainty_penalty: float = 1.0,
    top_k_per_hypothesis: int = 4,
    diversity_weight: float = 0.25,
) -> list[PerturbationCandidate]:
    hyp_by_id = {h.hypothesis_id: h for h in hypotheses}

    for c in candidates:
        h = hyp_by_id[c.hypothesis_id]
        sg = subgoals[c.canonical_phase_id]
        degs, uncs = [], []
        for ctx in contexts:
            seg_ab = _local_segment_for_phase(ctx, c.canonical_phase_id)
            if seg_ab is None:
                continue
            a, b = seg_ab
            if b - a < 3:
                continue
            start_idx = int(a + c.start_fraction * (b - a))
            start_idx = min(start_idx, b - 2)
            end_idx = min(phase_end_index(ctx.seg, start_idx), len(ctx.struct_norm) - 1)
            horizon = end_idx - start_idx
            if horizon < 2:
                continue
            row = _fcm_row(
                ctx.struct_norm[start_idx], ctx.actions[start_idx],
                lambda_probe * c.direction, float(ctx.cids[start_idx]),
                c.start_fraction, horizon,
            )
            resid, unc = fcm.predict(row[None, :])
            deg = degradation_score(
                sg, resid[0], structural_names,
                h.target_feature_names, h.target_type,
            )
            degs.append(deg)
            uncs.append(float(unc[0]))
        if not degs:
            c.predicted_degradation = None
            c.screening_score = -np.inf
            continue
        c.predicted_degradation = float(np.mean(degs))
        c.predicted_uncertainty = float(np.mean(uncs))
        c.screening_score = c.predicted_degradation - uncertainty_penalty * c.predicted_uncertainty

    # per-hypothesis Top-K with greedy diversity on direction
    selected: list[PerturbationCandidate] = []
    for h in hypotheses:
        pool = [c for c in candidates
                if c.hypothesis_id == h.hypothesis_id and c.screening_score is not None
                and np.isfinite(c.screening_score)]
        pool.sort(key=lambda c: -c.screening_score)
        chosen: list[PerturbationCandidate] = []
        for c in pool:
            if len(chosen) >= top_k_per_hypothesis:
                break
            if chosen:
                sim = max(
                    float(np.dot(c.direction, o.direction)) for o in chosen
                    if o.action_subspace == c.action_subspace
                ) if any(o.action_subspace == c.action_subspace for o in chosen) else 0.0
                # skip near-duplicate directions unless score justifies it
                if sim > 0.98 and c.screening_score < chosen[-1].screening_score + 1e-6:
                    continue
            chosen.append(c)
        selected.extend(chosen)
    return selected
