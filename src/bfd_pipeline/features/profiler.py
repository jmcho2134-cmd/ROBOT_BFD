"""Stage 5 - Automatic Feature Profiling.

Computes, from signal shape alone (never the name), how each feature behaves and
how much it should contribute to phase structure. Cross-demo scores compare
features on a common normalized-progress grid so demos of different length align.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from bfd_pipeline.core.types import FeatureProfile, FeatureTrajectory

GRID = 50  # normalized-progress resample resolution

# structural_score weights. `binary` credits transition features (contact,
# gripper) that are strong phase markers but non-monotonic (net delta ~ 0).
DEFAULT_WEIGHTS = {
    "event": 0.12,
    "trend": 0.12,
    "plateau": 0.06,
    "changepoint": 0.18,
    "endpoint": 0.12,
    "direction": 0.10,
    "timing": 0.08,
    "binary": 0.17,
    "robust": 0.05,
}


def _resample(sig: np.ndarray, valid: np.ndarray, grid: int = GRID) -> np.ndarray | None:
    sig = sig[valid]
    if sig.size < 3:
        return None
    x = np.linspace(0.0, 1.0, sig.size)
    xg = np.linspace(0.0, 1.0, grid)
    return np.interp(xg, x, sig)


def _feature_matrix(fts: list[FeatureTrajectory], j: int) -> np.ndarray | None:
    rows = []
    for ft in fts:
        sig = ft.signal(use_normalized=True)[:, j]
        valid = ft.valid_mask[:, j]
        r = _resample(sig, valid)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    return np.asarray(rows)  # (D, GRID)


def _event_score(m: np.ndarray) -> float:
    rng = np.ptp(m)
    if rng < 1e-9:
        return 0.0
    jumps = np.abs(np.diff(m))
    return float(np.clip(jumps.max() / (rng + 1e-9), 0.0, 1.0))


def _binary_score(m: np.ndarray) -> float:
    lo, hi = m.min(), m.max()
    rng = hi - lo
    if rng < 1e-9:
        return 0.0
    band = 0.15 * rng
    near = (m < lo + band) | (m > hi - band)
    return float(np.mean(near))


def _trend_score(M: np.ndarray) -> float:
    xg = np.linspace(0.0, 1.0, M.shape[1])
    vals = []
    for row in M:
        if np.ptp(row) < 1e-9:
            vals.append(0.0)
            continue
        rho, _ = spearmanr(row, xg)
        vals.append(abs(rho) if np.isfinite(rho) else 0.0)
    return float(np.mean(vals))


def _plateau_score(m: np.ndarray) -> float:
    d = np.abs(np.diff(m))
    if d.max() < 1e-9:
        return 1.0
    thr = 0.1 * d.max()
    return float(np.mean(d < thr))


def _changepoint_score(m: np.ndarray) -> float:
    sse_const = float(np.sum((m - m.mean()) ** 2))
    if sse_const < 1e-9:
        return 0.0
    best = sse_const
    for k in range(2, len(m) - 2):
        left, right = m[:k], m[k:]
        sse = np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
        best = min(best, sse)
    return float(np.clip(1.0 - best / sse_const, 0.0, 1.0))


def _endpoint_consistency(M: np.ndarray) -> float:
    tail = M[:, int(0.8 * M.shape[1]):].mean(axis=1)
    spread = np.ptp(M) + 1e-9
    return float(np.clip(1.0 - tail.std() / spread, 0.0, 1.0))


def _direction_consistency(M: np.ndarray) -> float:
    deltas = M[:, -1] - M[:, 0]
    signs = np.sign(deltas)
    signs = signs[np.abs(deltas) > 0.1 * (np.ptp(M) + 1e-9)]
    if signs.size == 0:
        return 0.0
    return float(abs(signs.mean()))


def _timing_consistency(M: np.ndarray) -> float:
    locs = []
    for row in M:
        d = np.abs(np.diff(row))
        if d.max() < 1e-9:
            continue
        locs.append(np.argmax(d) / len(d))
    if len(locs) < 2:
        return 0.0
    return float(np.clip(1.0 - np.std(locs) / 0.5, 0.0, 1.0))


def _structural_from_parts(parts: dict, weights: dict) -> float:
    s = sum(weights[k] * parts[k] for k in weights)
    return float(np.clip(s, 0.0, 1.0))


def _parts(M: np.ndarray) -> dict:
    m = M.mean(axis=0)
    return {
        "event": _event_score(m),
        "trend": _trend_score(M),
        "plateau": _plateau_score(m),
        "changepoint": _changepoint_score(m),
        "endpoint": _endpoint_consistency(M),
        "direction": _direction_consistency(M),
        "timing": _timing_consistency(M),
        "binary": _binary_score(m),
    }


def profile_feature(
    M: np.ndarray,
    name: str,
    weights: dict = DEFAULT_WEIGHTS,
    bootstrap: int = 40,
    seed: int = 0,
) -> FeatureProfile:
    rng = np.random.default_rng(seed)
    parts = _parts(M)

    # noise robustness: recompute structural score under small noise, measure drift
    scale = 0.05 * (np.ptp(M) + 1e-9)
    drifts = []
    base_struct = _structural_from_parts({**parts, "robust": 1.0}, weights)
    for _ in range(8):
        Mn = M + rng.normal(0, scale, M.shape)
        p = _parts(Mn)
        drifts.append(abs(_structural_from_parts({**p, "robust": 1.0}, weights) - base_struct))
    noise_robust = float(np.exp(-5.0 * np.mean(drifts)))

    parts_full = {**parts, "robust": noise_robust}
    structural = _structural_from_parts(parts_full, weights)

    # uncertainty: bootstrap over demo subsets
    boot = []
    D = M.shape[0]
    if D >= 3 and bootstrap > 0:
        for _ in range(bootstrap):
            idx = rng.integers(0, D, D)
            p = _parts(M[idx])
            boot.append(_structural_from_parts({**p, "robust": noise_robust}, weights))
        uncertainty = float(np.std(boot))
    else:
        uncertainty = 0.0

    m = M.mean(axis=0)
    confidence = float(np.clip(1.0 - 2.0 * uncertainty, 0.0, 1.0))

    return FeatureProfile(
        name=name,
        binary_score=_binary_score(m),
        event_score=parts["event"],
        trend_score=parts["trend"],
        plateau_score=parts["plateau"],
        changepoint_score=parts["changepoint"],
        endpoint_consistency=parts["endpoint"],
        cross_demo_direction_consistency=parts["direction"],
        cross_demo_timing_consistency=parts["timing"],
        noise_robustness=noise_robust,
        structural_score=structural,
        structural_uncertainty=uncertainty,
        confidence=confidence,
    )


def profile_features(
    fts: list[FeatureTrajectory],
    weights: dict = DEFAULT_WEIGHTS,
    bootstrap: int = 40,
    seed: int = 0,
) -> dict[str, FeatureProfile]:
    names = fts[0].names
    profiles: dict[str, FeatureProfile] = {}
    for j, name in enumerate(names):
        M = _feature_matrix(fts, j)
        if M is None:
            continue
        profiles[name] = profile_feature(
            M, name, weights=weights, bootstrap=bootstrap, seed=seed + j
        )
    return profiles
