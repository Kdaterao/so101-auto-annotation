"""CPU-only displacement evidence from recorded end-effector poses.

No actions, image motion, IK or inferred joint calibration are used. Joint-only
datasets need a validated forward-kinematics conversion before using this module.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from robot101.data.utilities.annotation_tree import atomic_json, cached_json, fingerprint
from robot101.data.utilities.episode_helpers import _as_numpy, _load_lerobot

MOTION_VERSION = "recorded-ee-displacement-v1"


def displacement_windows(states, timestamps, names, start, end, *, window_seconds=0.5,
                         displacement_scale=0.005, max_gap_seconds=0.1, position_unit="native"):
    """Summarize the observed TCP trajectory strictly inside a caption node.

    Adjacent time windows share boundary positions, interpolated only between
    valid samples within this node. Excursion detects out-and-back movement
    that net displacement alone would miss. The score is not a probability.
    """
    q, times = np.asarray(states, float), np.asarray(timestamps, float)
    if q.ndim != 2 or times.shape != (len(q),) or len(names) != q.shape[1]:
        raise ValueError("State shape, coordinate names and timestamps must agree")
    if not 0 <= start < end <= len(q):
        raise ValueError("Motion bounds must lie within the source episode")
    for value in (window_seconds, displacement_scale, max_gap_seconds):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Motion window, displacement scale and maximum gap must be finite and positive")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    required = ["ee.x", "ee.y", "ee.z"]
    if not all(name in names for name in required):
        raise ValueError("Need recorded ee.x/ee.y/ee.z; raw joints require calibrated forward kinematics")
    xyz = q[start:end, [names.index(name) for name in required]]
    t = times[start:end]
    quat_names = [f"ee.q{axis}" for axis in ("x", "y", "z", "w")]
    quat = q[start:end, [names.index(n) for n in quat_names]] if all(n in names for n in quat_names) else None
    grip = q[start:end, names.index("ee.gripper")] if "ee.gripper" in names else None
    rows = []
    bounds = np.arange(t[0], t[-1], window_seconds).tolist() + [float(t[-1])]
    if len(bounds) == 1:
        rows.append(dict(start_frame=start, end_frame=end, start_seconds=float(t[0]),
                         end_seconds=float(t[-1]), status="unavailable", reason="insufficient_samples"))
    for lo, hi in zip(bounds, bounds[1:]):
        # Include bracketing samples, never from another subtask. This also
        # checks gaps that straddle the nominal half-second boundary.
        a = max(0, int(np.searchsorted(t, lo, side="right")) - 1)
        b = min(len(t), int(np.searchsorted(t, hi, side="left")) + 1)
        row = dict(start_frame=start + a, end_frame=start + b,
                   start_seconds=float(lo), end_seconds=float(hi), status="unavailable")
        if b - a < 2:
            row["reason"] = "insufficient_samples"
        elif not np.isfinite(xyz[a:b]).all():
            row["reason"] = "invalid_position_samples"
        elif np.any(np.diff(t[a:b]) > max_gap_seconds):
            row["reason"] = "timestamp_gap"
        else:
            inner = (t > lo) & (t < hi)
            endpoints = np.stack([np.interp([lo, hi], t[a:b], xyz[a:b, dim]) for dim in range(3)], axis=1)
            positions = np.vstack([endpoints[0], xyz[inner], endpoints[1]])
            net = float(np.linalg.norm(positions[-1] - positions[0]))
            path = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            excursion = float(np.linalg.norm(positions[:, None] - positions[None, :], axis=-1).max())
            row.update(status="ok", net_displacement=net, max_excursion=excursion,
                       path_length=path, mean_path_speed=path / (hi - lo),
                       movement_score=min(1.0, excursion / displacement_scale))
            # Translation alone cannot rule out wrist rotation or jaw motion.
            if quat is not None:
                orientations = quat[a:b]
                norms = np.linalg.norm(orientations, axis=1)
                if np.isfinite(orientations).all() and np.all(norms > 1e-8):
                    normalized = orientations / norms[:, None]
                    dot = np.clip(np.abs(normalized @ normalized.T), 0, 1)
                    row["orientation_excursion_degrees"] = float(np.degrees(2 * np.arccos(dot.min())))
                else:
                    row["orientation_status"] = "invalid_quaternion_samples"
            if grip is not None and np.isfinite(grip[a:b]).all():
                row["gripper_range_native_units"] = float(np.ptp(grip[a:b]))
        rows.append(row)
    known = [row for row in rows if row["status"] == "ok"]
    return dict(status="ok" if known else "unavailable", method="recorded_end_effector_position",
                bounds="half_open_source_frames", start_frame=start, end_frame=end,
                position_unit=position_unit, window_seconds=window_seconds,
                displacement_scale=displacement_scale, max_gap_seconds=max_gap_seconds,
                score_definition="clip(max_pairwise_position_excursion / displacement_scale, 0, 1); not a probability",
                known_windows=len(known), unknown_windows=len(rows) - len(known), windows=rows,
                caveats="Measures TCP translation, not every arm link; low translation does not rule out rotation, jaw motion, contact, holding or release. Path length can include sensor noise. Native units are not assumed to be meters.")


def attach_motion_profiles(record, targets, output_dir, config):
    """Load pinned source state parquet only; cache profiles independently of APIs."""
    source = dict(repo_id=record["source_repo_id"], revision=record["source_revision"],
                  episode_index=record["episode_index"])
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", str(source["revision"])):
        raise ValueError("Motion evidence requires the caption source's immutable revision SHA")
    nodes = [node for node in record["nodes"] if node["id"] in targets]
    key = fingerprint(dict(version=MOTION_VERSION, source=source, config=config,
                           intervals=[(n["id"], n["start_frame"], n["end_frame"]) for n in nodes]))
    path = Path(output_dir) / "motion_profiles" / f'ep{record["episode_index"]:06d}.json'
    # Immutable compatible measurement caches can be reused after a dry run;
    # paid API run/response caches retain their separate --resume guards.
    artifact = cached_json(path, key, True)
    if artifact is None:
        dataset_class, _, _, cache_root = _load_lerobot()
        # Isolate revisions: later dataset releases may trim/reindex episodes.
        root = cache_root / "robot101_motion" / fingerprint(
            dict(repo_id=source["repo_id"], revision=source["revision"]))[:24]
        kwargs = dict(root=root, revision=source["revision"], episodes=[source["episode_index"]],
                      download_videos=False)
        try:
            dataset = dataset_class(source["repo_id"], **kwargs)
        except ValueError as exc:
            if "corresponds to no data" not in str(exc):
                raise
            dataset = dataset_class(source["repo_id"], force_cache_sync=True, **kwargs)
        data = dataset.hf_dataset
        states = np.stack([_as_numpy(row["observation.state"]).reshape(-1) for row in data])
        times = np.array([float(_as_numpy(row["timestamp"]).item()) for row in data])
        if len(states) != record["n_frames"]:
            raise ValueError("Motion source frame count does not match caption episode")
        if record.get("timestamps") is not None and not np.allclose(times, record["timestamps"], atol=1e-4, rtol=0):
            raise ValueError("Motion source timestamps do not match caption episode")
        if any(int(_as_numpy(row["episode_index"]).item()) != source["episode_index"] or
               int(_as_numpy(row["frame_index"]).item()) != i for i, row in enumerate(data)):
            raise ValueError("Motion source episode identity or frame order mismatch")
        names = dataset.meta.features["observation.state"].get("names")
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("Source state must declare named end-effector coordinates")
        profiles = {node["id"]: displacement_windows(states, times, names,
                    node["start_frame"], node["end_frame"], **config) for node in nodes}
        artifact = dict(fingerprint=key, version=MOTION_VERSION, source=source, config=config, nodes=profiles)
        atomic_json(path, artifact)
    for node in nodes:
        node["motion_profile"] = dict(artifact["nodes"][node["id"]], source=source, version=MOTION_VERSION)
    print(f'Motion ep{record["episode_index"]}: {len(nodes)} subtask profiles, '
          f'{sum(len(p["windows"]) for p in artifact["nodes"].values())} windows; no video decoding', flush=True)
