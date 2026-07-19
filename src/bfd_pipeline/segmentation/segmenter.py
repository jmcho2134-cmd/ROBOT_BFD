"""Stage 6 - Robust multi-demo phase segmentation.

Per-demo dynamic-programming segmentation over an evidence-derived candidate
boundary pool, followed by cross-demo canonical alignment and lightweight
stability checks. Phase labels are z0,z1,... (no semantic names).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from bfd_pipeline.core.types import (
    ArtifactMetadata,
    CanonicalPhaseModel,
    FeatureProfile,
    FeatureTrajectory,
    PhaseSegmentation,
    SegmentDescriptor,
)
from bfd_pipeline.segmentation.evidence import combined_evidence


def _structural_columns(ft: FeatureTrajectory, names: list[str]) -> np.ndarray:
    idx = [ft.names.index(n) for n in names]
    return ft.signal(use_normalized=True)[:, idx]


def _fit_cost(seg: np.ndarray) -> float:
    """Mean-over-features best-of {constant, linear} residual SSE for a segment."""
    L = len(seg)
    if L <= 2:
        return 0.0
    t = np.arange(L)
    total = 0.0
    for f in range(seg.shape[1]):
        y = seg[:, f]
        const = float(np.sum((y - y.mean()) ** 2))
        a, b = np.polyfit(t, y, 1)
        lin = float(np.sum((y - (a * t + b)) ** 2))
        total += min(const, lin)
    return total / seg.shape[1]


def _candidate_pool(
    E: np.ndarray,
    per_feature: np.ndarray,
    min_gap: int,
    edge_margin: int,
) -> list[int]:
    """Union of combined-evidence peaks and each structural feature's own peaks.

    Including per-feature peaks lets gradual-but-consistent transitions (e.g. a
    slow object-goal-distance decrease) offer a boundary even when they never
    form a sharp *combined* peak.
    """
    T = len(E)
    cand = set()
    peaks, _ = find_peaks(E, distance=max(1, min_gap // 2), prominence=0.03)
    cand.update(int(p) for p in peaks)
    for row in per_feature:
        fp, _ = find_peaks(row, distance=max(1, min_gap), prominence=0.15)
        cand.update(int(p) for p in fp)
    cand = {p for p in cand if edge_margin <= p <= T - edge_margin}
    return sorted({0} | cand | {T})


def _dp_segment(
    sig: np.ndarray,
    E: np.ndarray,
    candidates: list[int],
    min_len: int,
    beta: float,
    evidence_weight: float,
) -> tuple[list[int], float]:
    m = len(candidates)
    dp = [np.inf] * m
    back = [-1] * m
    dp[0] = 0.0
    for j in range(1, m):
        cj = candidates[j]
        for i in range(j):
            ci = candidates[i]
            if cj - ci < min_len:
                continue
            cost = dp[i] + _fit_cost(sig[ci:cj]) + beta
            if i > 0:
                cost -= evidence_weight * E[ci]
            if cost < dp[j]:
                dp[j] = cost
                back[j] = i
    # backtrack
    boundaries = []
    j = m - 1
    while j > 0:
        boundaries.append(candidates[j])
        j = back[j]
    boundaries.append(0)
    boundaries = sorted(set(boundaries))
    return boundaries, float(dp[m - 1])


def segment_demo(
    ft: FeatureTrajectory,
    profiles: dict[str, FeatureProfile],
    structural_names: list[str],
    windows_sec: list[float],
    min_phase_sec: float = 0.35,
    edge_margin_sec: float = 0.15,
    beta: float = 0.6,
    evidence_weight: float = 0.6,
    max_phase_count: int = 10,
) -> PhaseSegmentation:
    E, per_feature, _ = combined_evidence(ft, profiles, structural_names, windows_sec)
    dt = ft.dt
    T = ft.raw_values.shape[0]
    min_len = max(2, int(round(min_phase_sec / dt)))
    edge = max(1, int(round(edge_margin_sec / dt)))

    candidates = _candidate_pool(E, per_feature, min_gap=min_len, edge_margin=edge)
    boundaries, obj = _dp_segment(
        _structural_columns(ft, structural_names), E, candidates,
        min_len, beta, evidence_weight,
    )

    # enforce max_phase_count by dropping lowest-evidence internal boundaries
    while len(boundaries) - 1 > max_phase_count:
        internal = boundaries[1:-1]
        drop = min(internal, key=lambda b: E[b])
        boundaries.remove(drop)

    phase_ids = np.zeros(T, dtype=int)
    for k in range(len(boundaries) - 1):
        phase_ids[boundaries[k]:boundaries[k + 1]] = k
    b_scores = np.array([E[min(b, T - 1)] for b in boundaries])

    return PhaseSegmentation(
        episode_id=ft.episode_id,
        boundaries=boundaries,
        phase_ids=phase_ids,
        boundary_scores=b_scores,
        objective_value=obj,
        metadata=ArtifactMetadata(source_artifact_ids=[ft.episode_id]),
    )


def segment_descriptors(
    ft: FeatureTrajectory,
    seg: PhaseSegmentation,
    structural_names: list[str],
) -> list[SegmentDescriptor]:
    sig = _structural_columns(ft, structural_names)
    T = len(sig)
    out = []
    for k in range(seg.num_phases):
        a, b = seg.boundaries[k], seg.boundaries[k + 1]
        chunk = sig[a:b]
        t = np.arange(len(chunk))
        slope = np.array([
            np.polyfit(t, chunk[:, f], 1)[0] if len(chunk) > 1 else 0.0
            for f in range(chunk.shape[1])
        ])
        out.append(SegmentDescriptor(
            episode_id=ft.episode_id,
            local_phase_id=k,
            start=a, end=b,
            mean=chunk.mean(axis=0),
            std=chunk.std(axis=0),
            delta=chunk[-1] - chunk[0],
            slope=slope,
            endpoint=chunk[-1],
            duration_normalized=(b - a) / T,
        ))
    return out


def _descriptor_vec(d: SegmentDescriptor) -> np.ndarray:
    mid = (d.start + d.end) / 2.0
    return np.concatenate([d.mean, d.delta, [d.duration_normalized]])


def _align_to_canonical(
    demo_descs: list[SegmentDescriptor], centers: np.ndarray
) -> list[int]:
    """Monotonic assignment of local segments to canonical phase ids (allows merge)."""
    L = len(demo_descs)
    K = len(centers)
    vecs = np.stack([_descriptor_vec(d) for d in demo_descs])
    D = np.linalg.norm(vecs[:, None, :] - centers[None, :, :], axis=2)  # (L, K)
    INF = 1e18
    dp = np.full((L + 1, K + 1), INF)
    dp[0, 0] = 0.0
    bk = np.zeros((L + 1, K + 1), dtype=int)  # 0 stay, 1 advance
    for i in range(1, L + 1):
        for k in range(1, K + 1):
            stay = dp[i - 1, k] + D[i - 1, k - 1]      # merge into same canonical
            adv = dp[i - 1, k - 1] + D[i - 1, k - 1]   # start new canonical
            if stay <= adv:
                dp[i, k], bk[i, k] = stay, 0
            else:
                dp[i, k], bk[i, k] = adv, 1
    # backtrack
    path = [0] * L
    i, k = L, int(np.argmin(dp[L, 1:]) + 1)
    while i > 0:
        path[i - 1] = k - 1
        k = k - 1 if bk[i, k] == 1 else k
        i -= 1
    return path


def build_canonical(
    fts: list[FeatureTrajectory],
    segs: list[PhaseSegmentation],
    structural_names: list[str],
) -> CanonicalPhaseModel:
    counts = [s.num_phases for s in segs]
    K = int(np.median(counts))
    K = max(2, K)

    all_descs = {
        ft.episode_id: segment_descriptors(ft, s, structural_names)
        for ft, s in zip(fts, segs)
    }
    # medoid: a demo with K phases and lowest objective
    medoid_id = None
    best_obj = np.inf
    for ft, s in zip(fts, segs):
        if s.num_phases == K and s.objective_value < best_obj:
            best_obj, medoid_id = s.objective_value, ft.episode_id
    if medoid_id is None:  # fallback: closest count
        medoid_id = min(segs, key=lambda s: abs(s.num_phases - K)).episode_id

    centers = np.stack([_descriptor_vec(d) for d in all_descs[medoid_id]])
    if len(centers) != K:  # trim/pad
        centers = centers[:K] if len(centers) > K else np.pad(
            centers, ((0, K - len(centers)), (0, 0)), mode="edge")

    # one refinement pass: average aligned local descriptors per canonical id
    paths = {eid: _align_to_canonical(d, centers) for eid, d in all_descs.items()}
    dim = centers.shape[1]
    acc = [np.zeros(dim) for _ in range(K)]
    cnt = [0] * K
    for eid, descs in all_descs.items():
        for d, cid in zip(descs, paths[eid]):
            acc[cid] += _descriptor_vec(d)
            cnt[cid] += 1
    centers = np.stack([
        acc[k] / cnt[k] if cnt[k] > 0 else centers[k] for k in range(K)
    ])
    paths = {eid: _align_to_canonical(d, centers) for eid, d in all_descs.items()}

    # confidence: fraction of demos whose local phase count is within 1 of K
    close = np.mean([abs(c - K) <= 1 for c in counts])
    scales = np.stack([
        np.std([
            _descriptor_vec(all_descs[eid][i])
            for eid in all_descs for i, cid in enumerate(paths[eid]) if cid == k
        ] or [np.zeros(dim)], axis=0)
        for k in range(K)
    ])

    return CanonicalPhaseModel(
        canonical_phase_count=K,
        canonical_labels=[f"z{k}" for k in range(K)],
        phase_descriptor_centers=centers,
        phase_descriptor_scales=scales,
        demo_alignment_paths=paths,
        confidence=float(close),
    )
