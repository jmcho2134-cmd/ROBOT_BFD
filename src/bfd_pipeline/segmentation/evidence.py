"""Stage 6, Steps 1-2 - per-feature boundary evidence and reliability-weighted combine."""

from __future__ import annotations

import numpy as np

from bfd_pipeline.core.types import FeatureProfile, FeatureTrajectory


def _robust_mean_shift(x: np.ndarray, w: int) -> np.ndarray:
    """|mean(after) - mean(before)| in a +/- w window at each t."""
    T = len(x)
    out = np.zeros(T)
    for t in range(T):
        a0, a1 = max(0, t - w), t
        b0, b1 = t, min(T, t + w)
        if a1 - a0 < 2 or b1 - b0 < 2:
            continue
        out[t] = abs(np.median(x[b0:b1]) - np.median(x[a0:a1]))
    return out


def _slope_change(x: np.ndarray, w: int) -> np.ndarray:
    T = len(x)
    out = np.zeros(T)
    for t in range(T):
        a0, a1 = max(0, t - w), t
        b0, b1 = t, min(T, t + w)
        if a1 - a0 < 2 or b1 - b0 < 2:
            continue
        sl_before = np.polyfit(np.arange(a1 - a0), x[a0:a1], 1)[0]
        sl_after = np.polyfit(np.arange(b1 - b0), x[b0:b1], 1)[0]
        out[t] = abs(sl_after - sl_before)
    return out


def _variance_change(x: np.ndarray, w: int) -> np.ndarray:
    T = len(x)
    out = np.zeros(T)
    for t in range(T):
        a0, a1 = max(0, t - w), t
        b0, b1 = t, min(T, t + w)
        if a1 - a0 < 2 or b1 - b0 < 2:
            continue
        out[t] = abs(np.std(x[b0:b1]) - np.std(x[a0:a1]))
    return out


def _normalize(v: np.ndarray) -> np.ndarray:
    m = v.max()
    return v / m if m > 1e-9 else v


def feature_boundary_evidence(
    x: np.ndarray,
    windows: list[int],
) -> np.ndarray:
    """Multiscale evidence for one feature signal, normalized to [0,1]."""
    acc = np.zeros(len(x))
    for w in windows:
        acc = acc + _normalize(_robust_mean_shift(x, w))
        acc = acc + _normalize(_slope_change(x, w))
        acc = acc + _normalize(_variance_change(x, w))
    return _normalize(acc)


def combined_evidence(
    ft: FeatureTrajectory,
    profiles: dict[str, FeatureProfile],
    structural_names: list[str],
    windows_sec: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (E(t) (T,), per_feature_evidence (F_struct,T), reliability_weights).

    Evidence per structural feature is combined by a reliability-weighted mean
    with light single-feature-dominance trimming, so no lone feature dictates
    the boundary. Per-feature evidence is returned too so the candidate pool can
    include gradual transitions that never form a sharp *combined* peak.
    """
    dt = ft.dt
    windows = [max(1, int(round(s / dt))) for s in windows_sec]
    sig = ft.signal(use_normalized=True)

    ev_rows = []
    weights = []
    for name in structural_names:
        j = ft.names.index(name)
        ev = feature_boundary_evidence(sig[:, j], windows)
        ev_rows.append(ev)
        weights.append(profiles[name].reliability())

    if not ev_rows:
        z = np.zeros(ft.raw_values.shape[0])
        return z, np.zeros((0, len(z))), np.zeros(0)

    E = np.stack(ev_rows)          # (F_struct, T)
    w = np.asarray(weights)
    w = w / (w.sum() + 1e-9)

    weighted = E * w[:, None]
    combined = weighted.sum(axis=0)
    if E.shape[0] >= 3:
        combined = combined - 0.25 * weighted.max(axis=0)
    return _normalize(combined), E, w
