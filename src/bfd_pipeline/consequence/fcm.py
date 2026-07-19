"""Stage 9 - Residual Forward Consequence Model (screening model).

An ensemble of MLP regressors predicts the structural-feature residual from
(feature_t, action_t, perturbation, phase, progress, horizon). It is a *screening*
model: its job is to rank candidates so the simulator runs fewer rollouts. The
headline metric is Top-K recall, not MAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class FCMEnsemble:
    members: list
    x_scaler: StandardScaler
    n_targets: int

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs = self.x_scaler.transform(X)
        preds = np.stack([m.predict(Xs) for m in self.members])  # (M, N, F)
        return preds.mean(axis=0), preds.std(axis=0).mean(axis=1)


def train_fcm(
    X: np.ndarray,
    Y: np.ndarray,
    ensemble_size: int = 5,
    hidden=(128, 128),
    seed: int = 0,
) -> FCMEnsemble:
    x_scaler = StandardScaler().fit(X)
    Xs = x_scaler.transform(X)
    members = []
    for m in range(ensemble_size):
        reg = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=300,
            random_state=seed + m,
        )
        reg.fit(Xs, Y)
        members.append(reg)
    return FCMEnsemble(members=members, x_scaler=x_scaler, n_targets=Y.shape[1])


def evaluate_fcm(fcm: FCMEnsemble, X: np.ndarray, Y: np.ndarray) -> dict:
    mean, _ = fcm.predict(X)
    err = np.abs(mean - Y)
    ss_res = np.sum((Y - mean) ** 2, axis=0)
    var = np.var(Y, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    # R2 is meaningless for near-constant targets; report only informative ones.
    informative = var > 1e-4
    r2 = np.where(ss_tot > 1e-9, 1.0 - ss_res / np.maximum(ss_tot, 1e-9), 0.0)
    r2_info = r2[informative] if informative.any() else np.array([0.0])
    return {
        "mae": float(err.mean()),
        "r2_median_informative": float(np.median(r2_info)),
        "n_informative_features": int(informative.sum()),
        "r2_per_feature": np.clip(r2, -1.0, 1.0).tolist(),
    }


def top_k_recall(
    predicted_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int,
    m: int,
) -> float:
    """|TrueTopM ∩ PredictedTopK| / M."""
    true_top = set(np.argsort(-true_scores)[:m].tolist())
    pred_top = set(np.argsort(-predicted_scores)[:k].tolist())
    return len(true_top & pred_top) / max(1, m)
