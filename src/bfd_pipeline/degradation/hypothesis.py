"""Stage 10 - degradation hypothesis generation.

Turns each phase subgoal into feature-space degradation targets:
  * change_opposition: oppose a Change feature's demonstrated direction
  * hold_violation:    push a Hold feature off its reference distribution
Passive features never seed a hypothesis. Low-confidence subgoals are skipped.
"""

from __future__ import annotations

from bfd_pipeline.core.types import DegradationHypothesis, PhaseSubgoal


def generate_hypotheses(
    subgoals: dict[int, PhaseSubgoal],
    min_subgoal_confidence: float = 0.15,
    min_feature_importance: float = 0.5,
) -> list[DegradationHypothesis]:
    hyps: list[DegradationHypothesis] = []
    for k, sg in subgoals.items():
        if sg.confidence < min_subgoal_confidence:
            continue
        for cf in sg.change_features:
            if cf["importance"] < min_feature_importance:
                continue
            hyps.append(DegradationHypothesis(
                hypothesis_id=f"z{k}_chg_{cf['name']}",
                canonical_phase_id=k,
                target_type="change_opposition",
                target_feature_names=[cf["name"]],
                target_definition={
                    "demo_direction": cf["demo_direction"],
                    "median_delta": cf["median_delta"],
                },
                confidence=min(sg.confidence, cf["direction_consistency"]),
            ))
        for hf in sg.hold_features:
            if hf["importance"] < min_feature_importance:
                continue
            hyps.append(DegradationHypothesis(
                hypothesis_id=f"z{k}_hold_{hf['name']}",
                canonical_phase_id=k,
                target_type="hold_violation",
                target_feature_names=[hf["name"]],
                target_definition={
                    "reference_center": hf["reference_center"],
                    "reference_scale": hf["reference_scale"],
                },
                confidence=min(sg.confidence, hf["hold_consistency"]),
            ))
    return hyps
