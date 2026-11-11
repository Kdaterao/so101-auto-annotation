#!/usr/bin/env python3
"""Directly propagate reviewer-clicked POV points with BootsTAPIR.

This intentionally does not call the full preprocessing pipeline, cluster
POV features, or invoke Molmo/SAM. It reads accepted manual seeds from the
curation sidecars and tracks each seed only within its manually curated segment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from robot101.data.utilities.curation import (
    CURATION_DIR, POV_TRACK_DIR, SEGMENT_DIR, atomic_json, episode_filename,
    fingerprint, remote_episode_json, validate_curation,
)
from robot101.data.utilities.episode_helpers import _load_lerobot, _parse_episodes
from robot101.data.utilities.episode_io import decode_episode_cameras, load_episode_metadata
from robot101.paths import DEFAULT_CHECKPOINT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _dataset_for_episode(dataset_class, repo_id, revision, episode, cache_root, backend):
    root = Path(cache_root) / repo_id
    kwargs = dict(episodes=[episode], root=root, revision=revision, video_backend=backend)
    try:
        return dataset_class(repo_id, **kwargs)
    except ValueError as exc:
        if "corresponds to no data" not in str(exc):
            raise
        print(f"Episode {episode}: synchronizing pinned source shard", flush=True)
        return dataset_class(repo_id, force_cache_sync=True, **kwargs)


def _track_seed(tapir, frames, seed_offset: int, point_xy: np.ndarray, label: str):
    """Track one seed both ways and return [T,2] xy plus [T] visibility."""
    point = np.asarray(point_xy, dtype=np.float32).reshape(1, 2)
    before_frames = frames[:seed_offset + 1]
    after_frames = frames[seed_offset:]
    backward, backward_visible = tapir.track_segment_backward(
        before_frames, point, label=f"{label}/backward")
    forward, forward_visible, _, _ = tapir.track_video(
        after_frames, point, query_frame_index=0)
    # The seed frame occurs at the end of backward output and start of forward.
    tracks = np.concatenate((backward[0, :-1], forward[0]), axis=0)
    visible = np.concatenate((backward_visible[0, :-1], forward_visible[0]), axis=0)
    return tracks.astype(np.float32), visible.astype(bool)


def _write_episode(curation: dict, record: dict, source, data, tapir,
                   checkpoint_sha: str, camera: str, output_dir: Path,
                   resume: bool, curation_repo_id: str,
                   curation_revision: str) -> tuple[dict, Path]:
    episode = int(curation["episode_index"])
    filename = episode_filename(episode)
    base = output_dir / POV_TRACK_DIR
    meta_path = base / filename
    tracks_path = meta_path.with_suffix(".npz")
    source_revision = curation["source_revision"]
    source_repo = curation["source_repo_id"]
    input_fingerprint = fingerprint(dict(
        curation=fingerprint(curation), segmentation=curation["segment_record_fingerprint"],
        source_repo=source_repo, source_revision=source_revision,
        camera=camera, checkpoint_sha256=checkpoint_sha,
        tracker="BootsTAPIR/manual-seed-bidirectional-v1"))
    if resume and meta_path.is_file() and tracks_path.is_file():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == input_fingerprint:
            print(f"[ep {episode}] matching propagation already exists; skip", flush=True)
            return existing, tracks_path

    if data["episode_index"] != episode or data["n"] != curation["n_frames"]:
        raise ValueError(f"Episode {episode}: pinned source identity/frame count differs from curation")
    if not np.isclose(float(record["fps"]), float(curation["source_fps"]), atol=1e-6, rtol=0):
        raise ValueError(f"Episode {episode}: curated and segmented FPS differ")
    expected_ts = record.get("timestamps")
    if expected_ts is not None and not np.allclose(data["timestamps"], expected_ts, atol=1e-4, rtol=0):
        raise ValueError(f"Episode {episode}: source timestamps differ from segmentation")

    accepted = []
    for node_id, segment in curation["segments"].items():
        if segment.get("review_status") != "accepted":
            continue
        for point in segment.get("points", []):
            accepted.append((node_id, segment, point))
    if accepted:
        # Decode only the union of manually accepted segment intervals.
        frame_ids = sorted({frame for _, segment, _ in accepted
                            for frame in range(int(segment["start_frame"]), int(segment["end_frame"]))})
        decode_episode_cameras(source, data, [camera], indices=frame_ids,
                               batch_size=64, backend=data.get("video_backend", "pyav"))

    features: dict[str, np.ndarray] = {}
    entries = []
    for index, (node_id, segment, point) in enumerate(accepted):
        start, end = int(segment["start_frame"]), int(segment["end_frame"])
        seed = int(point["seed_frame"])
        if not start <= seed < end:
            raise ValueError(f"Episode {episode}, segment {node_id}: point seed lies outside segment")
        if point.get("camera", camera) != camera:
            raise ValueError(f"Episode {episode}, segment {node_id}: only {camera} POV seeds are supported")
        clip = data["frames"][camera][start:end]
        masks = np.asarray(data["masks"][camera][start:end], dtype=bool)
        if len(clip) != end - start or any(frame is None for frame in clip):
            raise ValueError(f"Episode {episode}, segment {node_id}: source video frames are missing")
        seed_offset = seed - start
        if not masks[seed_offset]:
            raise ValueError(f"Episode {episode}, segment {node_id}: seed frame has no valid POV image")
        seed_xy = np.asarray(point["xy"], dtype=np.float32).reshape(2)
        height, width = clip[seed_offset].shape[:2]
        if not (np.isfinite(seed_xy).all() and 0 <= seed_xy[0] < width and 0 <= seed_xy[1] < height):
            raise ValueError(f"Episode {episode}, segment {node_id}: seed coordinate is outside the POV image")
        point_id = str(point.get("id") or f"{node_id}-point-{index:04d}")
        try:
            tracks, visible = _track_seed(tapir, clip, seed_offset, seed_xy,
                                          f"ep{episode}/{node_id}/{point_id}")
            visible &= masks
            # Preserve the human click exactly at its seed frame.
            tracks[seed_offset] = seed_xy
            visible[seed_offset] = True
            coords_key, visible_key = f"xy_{index:04d}", f"visible_{index:04d}"
            features[coords_key] = tracks
            features[visible_key] = visible
            entries.append(dict(
                point_id=point_id, node_id=node_id, label=point["label"], camera=camera,
                interval_start=start, interval_end=end, seed_frame=seed,
                seed_xy=seed_xy.tolist(), image_size=[width, height],
                coordinates_key=coords_key, visibility_key=visible_key,
                status="ok" if bool(visible.any()) else "tracker_lost",
                visible_frames=int(visible.sum()), frame_count=end-start,
            ))
        except Exception as exc:
            entries.append(dict(
                point_id=point_id, node_id=node_id, label=point.get("label", ""), camera=camera,
                interval_start=start, interval_end=end, seed_frame=seed, seed_xy=seed_xy.tolist(),
                status="failed", error=f"{type(exc).__name__}: {exc}"))
            print(f"[ep {episode} {node_id} {point_id}] tracking failed: {exc}", file=sys.stderr, flush=True)

    if features:
        _save_npz_atomic(tracks_path, features)
    else:
        _save_npz_atomic(tracks_path, {"empty": np.zeros((0,), dtype=np.uint8)})
    output = dict(
        schema_version=1, episode_index=episode, source_repo_id=source_repo,
        source_revision=source_revision, curation_repo_id=curation_repo_id,
        curation_revision=curation_revision,
        segment_repo_id=curation["segment_repo_id"], segment_revision=curation["segment_revision"],
        segment_record_fingerprint=curation["segment_record_fingerprint"],
        curation_fingerprint=fingerprint(curation), camera=camera,
        tracking_method="manual_seed_bidirectional_bootstapir", checkpoint_sha256=checkpoint_sha,
        bounds="half_open_source_frames", fingerprint=input_fingerprint,
        status=("ok" if any(item["status"] == "ok" for item in entries)
                else "no_accepted_points" if not entries else "failed"),
        tracks_file=tracks_path.name, points=entries)
    atomic_json(meta_path, output)
    return output, tracks_path


def _write_failure(curation: dict, camera: str, checkpoint_sha: str,
                   curation_repo_id: str, curation_revision: str,
                   output_dir: Path, error: Exception) -> tuple[dict, Path]:
    episode = int(curation["episode_index"])
    base = output_dir / POV_TRACK_DIR
    meta_path = base / episode_filename(episode)
    tracks_path = meta_path.with_suffix(".npz")
    _save_npz_atomic(tracks_path, {"empty": np.zeros((0,), dtype=np.uint8)})
    output = dict(
        schema_version=1, episode_index=episode,
        source_repo_id=curation.get("source_repo_id"),
        source_revision=curation.get("source_revision"),
        curation_repo_id=curation_repo_id, curation_revision=curation_revision,
        segment_repo_id=curation.get("segment_repo_id"),
        segment_revision=curation.get("segment_revision"),
        segment_record_fingerprint=curation.get("segment_record_fingerprint"),
        curation_fingerprint=fingerprint(curation), camera=camera,
        tracking_method="manual_seed_bidirectional_bootstapir",
        checkpoint_sha256=checkpoint_sha, bounds="half_open_source_frames",
        status="failed", failure=f"{type(error).__name__}: {error}",
        tracks_file=tracks_path.name, points=[])
    atomic_json(meta_path, output)
    return output, tracks_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF dataset containing curation/episodes sidecars")
    parser.add_argument("--revision", default="main", help="Curation snapshot; resolved to an immutable SHA")
    parser.add_argument("--episodes", help="Optional episode IDs/ranges; default: all curated episodes")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pov_tracks"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None, help="TAPIR device; defaults to CUDA, MPS, then CPU")
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument("--camera", default="wrist", choices=["wrist"])
    parser.add_argument("--video-backend", choices=["pyav", "torchcodec"], default="pyav")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload sidecars under pov_tracks/episodes")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/so101-pov-propagation"))
    parser.add_argument("--source-cache-dir", type=Path, help="LeRobot source video cache root")
    args = parser.parse_args()
    if args.frame_batch_size < 1:
        parser.error("--frame-batch-size must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"TAPIR checkpoint not found: {args.checkpoint}")
    args.episodes = set(_parse_episodes(args.episodes)) if args.episodes else None
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        curation_revision = api.repo_info(args.repo_id, repo_type="dataset", revision=args.revision).sha
        files = set(api.list_repo_files(args.repo_id, repo_type="dataset", revision=curation_revision))
        names = sorted(name for name in files if name.startswith(f"{CURATION_DIR}/ep") and name.endswith(".json"))
        if args.episodes is not None:
            names = [name for name in names
                     if int(Path(name).stem[2:]) in args.episodes]
            missing = args.episodes - {int(Path(name).stem[2:]) for name in names}
            if missing:
                raise ValueError(f"Curated episodes are missing: {sorted(missing)}")
        if not names:
            raise ValueError(f"No curation sidecars found under {CURATION_DIR}/")
        LeRobotDataset, _, _, cache_root = _load_lerobot()
        if args.source_cache_dir is not None:
            cache_root = args.source_cache_dir
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_sha = _sha256(args.checkpoint)
        tapir = None
        from robot101.perception.tracking import BootsTAPIR
        source_datasets = {}
        started = perf_counter()
        try:
            for name in names:
                episode = int(Path(name).stem[2:])
                curation = None
                try:
                    curation = remote_episode_json(args.repo_id, curation_revision, name, args.cache_dir)
                    if curation.get("episode_review_status") != "accepted":
                        print(f"[ep {episode}] episode status={curation.get('episode_review_status')}; skip", flush=True)
                        continue
                    segment_repo = curation.get("segment_repo_id") or args.repo_id
                    segment_revision = curation.get("segment_revision")
                    if not segment_revision:
                        raise ValueError(f"Episode {episode}: curation sidecar has no pinned segment revision")
                    segment_name = f"{SEGMENT_DIR}/{episode_filename(episode)}"
                    record = remote_episode_json(segment_repo, segment_revision, segment_name, args.cache_dir)
                    validate_curation(curation, record)
                    source_repo, source_revision = curation["source_repo_id"], curation["source_revision"]
                    if not source_revision:
                        raise ValueError(f"Episode {episode}: source revision is not pinned")
                    source_key = source_repo, source_revision, episode
                    source = source_datasets.get(source_key)
                    if source is None:
                        source = _dataset_for_episode(LeRobotDataset, source_repo, source_revision,
                                                      episode, cache_root, args.video_backend)
                        source_datasets[source_key] = source
                    data = load_episode_metadata(source, [args.camera], preserve_auxiliary=False)
                    data["video_backend"] = args.video_backend
                    has_seeds = any(segment.get("review_status") == "accepted" and segment.get("points")
                                    for segment in curation["segments"].values())
                    if has_seeds and tapir is None:
                        tapir = BootsTAPIR(checkpoint=args.checkpoint, device=args.device,
                                           frame_batch_size=args.frame_batch_size)
                    output, track_path = _write_episode(
                        curation, record, source, data, tapir, checkpoint_sha, args.camera,
                        args.output_dir, args.resume, args.repo_id, curation_revision)
                    print(f"[ep {episode}] {output['status']}: {len(output['points'])} point tracks", flush=True)
                except Exception as exc:
                    if curation is None:
                        print(f"[ep {episode}] unable to load curation: {exc}", file=sys.stderr, flush=True)
                        continue
                    output, track_path = _write_failure(curation, args.camera, checkpoint_sha,
                                                        args.repo_id, curation_revision,
                                                        args.output_dir, exc)
                    print(f"[ep {episode}] failed: {exc}", file=sys.stderr, flush=True)
                if args.push_to_hub:
                    api.upload_file(path_or_fileobj=str(track_path),
                                    path_in_repo=f"{POV_TRACK_DIR}/{track_path.name}",
                                    repo_id=args.repo_id, repo_type="dataset",
                                    commit_message=f"Propagate POV points for episode {episode}")
                    api.upload_file(path_or_fileobj=str(args.output_dir / POV_TRACK_DIR / episode_filename(episode)),
                                    path_in_repo=f"{POV_TRACK_DIR}/{episode_filename(episode)}",
                                    repo_id=args.repo_id, repo_type="dataset",
                                    commit_message=f"Record POV tracking metadata for episode {episode}")
        finally:
            if tapir is not None:
                del tapir
        print(f"Completed POV propagation in {perf_counter()-started:.1f}s", flush=True)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
