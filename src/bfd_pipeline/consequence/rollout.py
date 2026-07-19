"""Stage 8 - perturbation rollout dataset.

Branches the synthetic env from a demo state: one branch runs demo actions, the
other adds a perturbation over a short window then returns to demo actions. The
structural-feature residual (pert - demo) at the phase endpoint is the FCM target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bfd_pipeline.core.types import FeatureTrajectory
from bfd_pipeline.envs.synthetic_env import (
    ACTION_DIM,
    SyntheticPickPlaceEnv,
    raw_from_states,
)
from bfd_pipeline.features.engine import compute_feature_trajectory
from bfd_pipeline.features.normalization import RobustScaler
from bfd_pipeline.features.registry import FeatureRegistry


def canonical_phase_ids(seg, canonical, episode_id) -> np.ndarray:
    """Map per-timestep local phase ids to canonical ids via the alignment path."""
    path = canonical.demo_alignment_paths.get(
        episode_id, list(range(seg.num_phases))
    )
    out = np.zeros_like(seg.phase_ids)
    for local_k in range(seg.num_phases):
        cid = path[local_k] if local_k < len(path) else path[-1]
        out[seg.phase_ids == local_k] = cid
    return out


def phase_end_index(seg, t: int) -> int:
    for k in range(seg.num_phases):
        if seg.boundaries[k] <= t < seg.boundaries[k + 1]:
            return seg.boundaries[k + 1]
    return seg.boundaries[-1]


def generate_directions(layout: dict, rng, n_random: int = 4):
    dirs = []
    for name, idxs in (("position", layout["position"]),
                       ("rotation", layout["rotation"]),
                       ("gripper", layout["gripper"])):
        for i in idxs:
            for sign in (+1.0, -1.0):
                d = np.zeros(ACTION_DIM)
                d[i] = sign
                dirs.append((name, d))
        for _ in range(n_random if name != "gripper" else 0):
            d = np.zeros(ACTION_DIM)
            v = rng.normal(0, 1, len(idxs))
            d[idxs] = v / (np.linalg.norm(v) + 1e-9)
            dirs.append((name, d))
    return dirs


def _features_struct(
    states, actions, cf, reg, scaler, structural_idx,
) -> np.ndarray:
    raw = raw_from_states("branch", states, actions, cf)
    ft = compute_feature_trajectory(raw, reg, scaler=scaler)
    return ft.normalized_values[:, structural_idx]


def branch_states(env, start_state, actions, start_idx, window_len, dir_full, lam, n_steps):
    a = np.array(actions[start_idx:start_idx + n_steps], dtype=float)
    if len(a) == 0:
        return None
    w = min(window_len, len(a))
    a[:w] = np.clip(a[:w] + lam * dir_full, -1.0, 1.0)
    return env.rollout(start_state, a)


@dataclass
class FCMDataset:
    X: np.ndarray
    Y: np.ndarray
    feature_cols: int


def build_fcm_dataset(
    episodes,
    fts: list[FeatureTrajectory],
    segs,
    canonical,
    scaler: RobustScaler,
    reg: FeatureRegistry,
    structural_names: list[str],
    layout: dict,
    control_frequency: float = 20.0,
    lambdas=(0.0, 0.15, 0.30, 0.5),
    window_len: int = 3,
    starts_per_phase: int = 2,
    max_demos: int = 12,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    structural_idx = [reg.names.index(n) for n in structural_names]
    env = SyntheticPickPlaceEnv()
    X_rows, Y_rows = [], []

    ep_by_id = {e.episode_id: e for e in episodes}
    directions = generate_directions(layout, rng, n_random=2)

    for ft, seg in list(zip(fts, segs))[:max_demos]:
        ep = ep_by_id[ft.episode_id]
        states = ep.states
        actions = ep.actions
        cids = canonical_phase_ids(seg, canonical, ft.episode_id)
        struct_norm = ft.normalized_values[:, structural_idx]

        for k in range(seg.num_phases):
            a, b = seg.boundaries[k], seg.boundaries[k + 1]
            if b - a < 3:
                continue
            starts = rng.integers(a, max(a + 1, b - 1), size=starts_per_phase)
            for start_idx in np.unique(starts):
                start_idx = int(start_idx)
                # clamp endpoint to a recorded demo index so the branch never
                # steps beyond states the demo actually has (Gate 8 exactness).
                end_idx = min(phase_end_index(seg, start_idx), len(states) - 1)
                horizon = end_idx - start_idx
                if horizon < 2:
                    continue
                demo_end = struct_norm[end_idx]
                for subspace, d in directions:
                    for lam in lambdas:
                        st = branch_states(env, states[start_idx], actions,
                                           int(start_idx), window_len, d, lam, horizon)
                        if st is None:
                            continue
                        used_actions = actions[int(start_idx):int(start_idx) + horizon]
                        pf = _features_struct(st, used_actions, control_frequency,
                                              reg, scaler, structural_idx)
                        residual = pf[-1] - demo_end
                        cid = int(cids[int(start_idx)])
                        progress = (int(start_idx) - a) / max(1, b - a)
                        feat_t = struct_norm[int(start_idx)]
                        act_t = actions[int(start_idx)]
                        row = np.concatenate([
                            feat_t, act_t, lam * d,
                            [cid], [progress], [horizon],
                        ])
                        X_rows.append(row)
                        Y_rows.append(residual)

    X = np.asarray(X_rows)
    Y = np.asarray(Y_rows)
    return FCMDataset(X=X, Y=Y, feature_cols=len(structural_names))
