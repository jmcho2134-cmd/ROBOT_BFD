"""Stage 7 - Phase subgoal inference + completion / degradation scoring.

Per canonical phase, decides which structural features must Change (consistent
signed delta across demos) vs Hold (stable endpoint), and defines a completion
score. The degradation score used by Stages 10-12 is a signed function of the
feature residual against these demonstrated directions.
"""

from __future__ import annotations

import numpy as np

from bfd_pipeline.core.types import (
    CanonicalPhaseModel,
    FeatureProfile,
    FeatureTrajectory,
    PhaseSubgoal,
)
from bfd_pipeline.segmentation.segmenter import segment_descriptors

EPS = 1e-6


def _phase_groups(fts, segs, canonical, structural_names):
    """canonical_id -> list of (start_vec, end_vec, within_std_vec) over structural feats."""
    groups: dict[int, list[tuple]] = {k: [] for k in range(canonical.canonical_phase_count)}
    for ft, seg in zip(fts, segs):
        descs = segment_descriptors(ft, seg, structural_names)
        path = canonical.demo_alignment_paths.get(ft.episode_id, list(range(len(descs))))
        for d, cid in zip(descs, path):
            if cid < 0 or cid >= canonical.canonical_phase_count:
                continue
            start_vec = d.mean - d.delta / 2.0  # approx start; refined below
            groups[cid].append((d.endpoint - d.delta, d.endpoint, d.std))
    return groups


def infer_subgoals(
    fts: list[FeatureTrajectory],
    segs,
    canonical: CanonicalPhaseModel,
    profiles: dict[str, FeatureProfile],
    structural_names: list[str],
    min_change_effect: float = 0.5,
    min_direction_consistency: float = 0.6,
    min_hold_consistency: float = 0.6,
) -> dict[int, PhaseSubgoal]:
    groups = _phase_groups(fts, segs, canonical, structural_names)
    subgoals: dict[int, PhaseSubgoal] = {}

    for k in range(canonical.canonical_phase_count):
        entries = groups[k]
        if not entries:
            continue
        starts = np.stack([e[0] for e in entries])   # (n, F)
        ends = np.stack([e[1] for e in entries])
        within = np.stack([e[2] for e in entries])
        deltas = ends - starts

        change_features, hold_features, passive = [], [], []
        start_center = np.median(starts, axis=0)
        end_center = np.median(ends, axis=0)

        for f, name in enumerate(structural_names):
            md = float(np.median(deltas[:, f]))
            sig = np.sign(deltas[:, f])
            sig = sig[np.abs(deltas[:, f]) > 0.1]
            dir_consistency = float(abs(sig.mean())) if sig.size else 0.0
            importance = profiles[name].structural_score

            if abs(md) >= min_change_effect and dir_consistency >= min_direction_consistency:
                change_features.append({
                    "name": name,
                    "demo_direction": int(np.sign(md)),
                    "median_delta": md,
                    "start_center": float(start_center[f]),
                    "end_center": float(end_center[f]),
                    "direction_consistency": dir_consistency,
                    "importance": importance,
                })
            else:
                endpoint_std = float(np.std(ends[:, f]))
                mean_within = float(np.mean(within[:, f]))
                hold_consistency = float(np.exp(-(endpoint_std + mean_within)))
                if hold_consistency >= min_hold_consistency:
                    hold_features.append({
                        "name": name,
                        "reference_center": float(end_center[f]),
                        "reference_scale": max(endpoint_std, 0.1),
                        "hold_consistency": hold_consistency,
                        "importance": importance,
                    })
                else:
                    passive.append(name)

        # non-structural features are passive by construction
        conf = float(np.clip(
            (len(change_features) + len(hold_features))
            / max(1, len(structural_names)), 0.0, 1.0
        ))
        subgoals[k] = PhaseSubgoal(
            canonical_phase_id=k,
            change_features=change_features,
            hold_features=hold_features,
            passive_features=passive,
            endpoint_center=end_center,
            endpoint_scale=np.std(ends, axis=0),
            confidence=conf,
        )
    return subgoals


def completion_score(
    subgoal: PhaseSubgoal,
    x_struct: np.ndarray,
    structural_names: list[str],
) -> float:
    """Demonstrated-subgoal completion at a structural feature endpoint vector."""
    num, den = 0.0, 0.0
    idx = {n: i for i, n in enumerate(structural_names)}
    for cf in subgoal.change_features:
        f = idx[cf["name"]]
        s, e = cf["start_center"], cf["end_center"]
        denom = e - s
        if abs(denom) < EPS:
            continue
        prog = np.clip((x_struct[f] - s) / denom, -0.2, 1.2)
        w = cf["importance"]
        num += w * prog
        den += w
    for hf in subgoal.hold_features:
        f = idx[hf["name"]]
        hold = float(np.exp(-abs(x_struct[f] - hf["reference_center"])
                            / max(hf["reference_scale"], EPS)))
        w = hf["importance"]
        num += w * hold
        den += w
    return float(num / den) if den > 0 else 0.0


def degradation_score(
    subgoal: PhaseSubgoal,
    residual_struct: np.ndarray,
    structural_names: list[str],
    target_feature_names: list[str],
    target_type: str,
) -> float:
    """Signed degradation from a structural-feature residual (pert - demo).

    change_opposition: reward residual that moves target features *against*
    their demonstrated direction. hold_violation: reward any departure.
    """
    idx = {n: i for i, n in enumerate(structural_names)}
    dir_by_name = {cf["name"]: cf["demo_direction"] for cf in subgoal.change_features}
    score = 0.0
    for name in target_feature_names:
        if name not in idx:
            continue
        r = residual_struct[idx[name]]
        if target_type == "change_opposition":
            score += -dir_by_name.get(name, 0) * r
        else:  # hold_violation
            score += abs(r)
    return float(score)
