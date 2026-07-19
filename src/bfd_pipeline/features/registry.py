"""Stage 4 - feature registry.

Enforces Gate B (Feature Function Only): the registry stores a name and a
callable, and nothing else. Any attempt to attach role/direction/boundary
metadata is rejected here so the "feature-function-only" invariant is checked
in code, not just by convention.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from bfd_pipeline.core.types import RawTrajectory
from bfd_pipeline.features.feature_functions import DEFAULT_FEATURE_FUNCTIONS

FeatureFunction = Callable[[RawTrajectory], np.ndarray]

# Substrings that would indicate someone is smuggling semantics into a name.
_FORBIDDEN_NAME_TOKENS = (
    "progress",
    "event",
    "boundary",
    "higher_is",
    "worse",
    "better",
    "reward",
)


class FeatureRegistry:
    def __init__(self) -> None:
        self._functions: dict[str, FeatureFunction] = {}

    def register(self, name: str, fn: FeatureFunction) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("feature name must be a non-empty string")
        if name in self._functions:
            raise ValueError(f"feature '{name}' already registered")
        lowered = name.lower()
        for token in _FORBIDDEN_NAME_TOKENS:
            if token in lowered:
                raise ValueError(
                    f"feature name '{name}' encodes semantics ('{token}'); "
                    "roles/directions must be learned, not named (Gate B)"
                )
        if not callable(fn):
            raise ValueError(f"feature '{name}' must map to a callable")
        self._functions[name] = fn

    @property
    def names(self) -> list[str]:
        """Feature order = insertion order, fixed for the whole pipeline."""
        return list(self._functions.keys())

    def function(self, name: str) -> FeatureFunction:
        return self._functions[name]

    def items(self):
        return self._functions.items()

    def __len__(self) -> int:
        return len(self._functions)


def default_registry() -> FeatureRegistry:
    reg = FeatureRegistry()
    for name, fn in DEFAULT_FEATURE_FUNCTIONS.items():
        reg.register(name, fn)
    return reg
