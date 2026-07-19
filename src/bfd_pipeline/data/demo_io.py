"""Stage 2 - Demo Loader.

Reads HDF5 demonstration files and normalizes their internal layout into
`EpisodeData`. Downstream stages never touch raw HDF5 structure.

Supported layouts (auto-detected):
  * robomimic / robosuite style:  data/demo_0/{states,actions,model_file}
  * flat single-episode:          {states,actions}
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from bfd_pipeline.core.types import ArtifactMetadata, EpisodeData


@dataclass
class LoadReport:
    accepted: list[str]
    rejected: dict[str, str]  # episode_id -> reason


def _to_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _find_episode_groups(f: h5py.File) -> list[tuple[str, h5py.Group]]:
    """Return (episode_id, group) pairs, handling both nested and flat layouts."""
    # robomimic style: top-level "data" group whose children are episodes.
    if "data" in f and isinstance(f["data"], h5py.Group):
        data = f["data"]
        groups = [
            (key, data[key])
            for key in data.keys()
            if isinstance(data[key], h5py.Group)
        ]
        if groups:
            return groups
    # any top-level group that itself holds states/actions
    groups = [
        (key, f[key])
        for key in f.keys()
        if isinstance(f[key], h5py.Group) and "actions" in f[key]
    ]
    if groups:
        return groups
    # flat single episode
    if "states" in f and "actions" in f:
        return [("demo_0", f)]  # type: ignore[list-item]
    return []


def _read_episode(
    episode_id: str,
    group,
    source_path: str,
    dtype: str = "float32",
) -> EpisodeData:
    if "states" not in group:
        raise KeyError("missing 'states'")
    if "actions" not in group:
        raise KeyError("missing 'actions'")

    states = np.asarray(group["states"][()], dtype=dtype)
    actions = np.asarray(group["actions"][()], dtype=dtype)

    model_xml = None
    for key in ("model_file", "model_xml", "model"):
        if key in group:
            model_xml = _to_str(group[key][()])
            break

    env_info: dict = {}
    if "env_args" in group.attrs:
        env_info["env_args"] = _to_str(group.attrs["env_args"])
    for attr_key in group.attrs:
        if attr_key not in env_info:
            env_info[attr_key] = group.attrs[attr_key]

    success = None
    if "success" in group.attrs:
        success = bool(group.attrs["success"])
    elif "successes" in group:
        success = bool(np.asarray(group["successes"][()]).any())

    episode = EpisodeData(
        episode_id=episode_id,
        states=states,
        actions=actions,
        model_xml=model_xml,
        env_info=env_info,
        # success unknown at load time is recomputed by the Stage 3 adapter;
        # default True keeps the episode in the pool until then.
        success=True if success is None else success,
        source_path=source_path,
        metadata=ArtifactMetadata(source_artifact_ids=[source_path]),
    )
    episode.validate()
    return episode


def load_demo_file(
    path: str,
    dtype: str = "float32",
) -> tuple[list[EpisodeData], LoadReport]:
    """Load every episode in one HDF5 file.

    Critical errors (missing/mismatched/NaN data) reject that episode only;
    the rest of the file still loads. Returns (episodes, report).
    """
    episodes: list[EpisodeData] = []
    accepted: list[str] = []
    rejected: dict[str, str] = {}

    with h5py.File(path, "r") as f:
        groups = _find_episode_groups(f)
        if not groups:
            raise ValueError(f"no episode groups found in {path}")
        for episode_id, group in groups:
            try:
                episode = _read_episode(episode_id, group, path, dtype=dtype)
            except (KeyError, ValueError) as exc:
                rejected[episode_id] = str(exc)
                continue
            episodes.append(episode)
            accepted.append(episode_id)

    return episodes, LoadReport(accepted=accepted, rejected=rejected)
