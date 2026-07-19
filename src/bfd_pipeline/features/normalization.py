"""Stage 4 - robust feature normalization.

center = median, scale = max(IQR / 1.349, epsilon), then clip. Statistics are
fit on TRAIN demos only (no validation/test leakage) and reused everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_IQR_TO_STD = 1.349


@dataclass
class RobustScaler:
    names: list[str]
    center: np.ndarray            # (F,)
    scale: np.ndarray             # (F,)
    epsilon: float = 1e-6
    clip_value: float = 10.0

    @classmethod
    def fit(
        cls,
        feature_matrices: list[np.ndarray],
        names: list[str],
        epsilon: float = 1e-6,
        clip_value: float = 10.0,
    ) -> "RobustScaler":
        """Fit on a list of (T_i, F) train matrices (only finite values used)."""
        stacked = np.concatenate(feature_matrices, axis=0)  # (sum T, F)
        if stacked.shape[1] != len(names):
            raise ValueError("feature dim does not match names length")
        center = np.full(stacked.shape[1], 0.0)
        scale = np.full(stacked.shape[1], epsilon)
        for j in range(stacked.shape[1]):
            col = stacked[:, j]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            median = float(np.median(col))
            q75, q25 = np.percentile(col, [75, 25])
            iqr = float(q75 - q25)
            center[j] = median
            scale[j] = max(iqr / _IQR_TO_STD, epsilon)
        return cls(
            names=list(names),
            center=center,
            scale=scale,
            epsilon=epsilon,
            clip_value=clip_value,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        z = (values - self.center) / self.scale
        return np.clip(z, -self.clip_value, self.clip_value)

    def to_dict(self) -> dict:
        return {
            "names": self.names,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "epsilon": self.epsilon,
            "clip_value": self.clip_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RobustScaler":
        return cls(
            names=list(d["names"]),
            center=np.asarray(d["center"], dtype=float),
            scale=np.asarray(d["scale"], dtype=float),
            epsilon=float(d["epsilon"]),
            clip_value=float(d["clip_value"]),
        )
