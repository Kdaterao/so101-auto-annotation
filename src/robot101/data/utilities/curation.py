"""Shared storage and validation for the manual episode curation workflow."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


CURATION_SCHEMA_VERSION = 1
CURATION_DIR = "curation/episodes"
SEGMENT_DIR = "captioning/episodes"
POV_TRACK_DIR = "pov_tracks/episodes"


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def episode_filename(episode: int) -> str:
    return f"ep{int(episode):06d}.json"


def fresh_curation(record: dict, repo_id: str, revision: str) -> dict:
    selected = set(record.get("selected_node_ids", []))
    nodes = {}
    for node in record.get("nodes", []):
        if node.get("id") not in selected:
            continue
        nodes[node["id"]] = {
            "start_frame": int(node["start_frame"]),
            "end_frame": int(node["end_frame"]),
            "text": "",
            "review_status": "todo",
            "points": [],
        }
    return {
        "schema_version": CURATION_SCHEMA_VERSION,
        "episode_index": int(record["episode_index"]),
        "source_repo_id": record["source_repo_id"],
        "source_revision": record["source_revision"],
        "source_fps": float(record["fps"]),
        "n_frames": int(record["n_frames"]),
        "bounds": "half_open_source_frames",
        "segment_repo_id": repo_id,
        "segment_revision": revision,
        "segment_record_fingerprint": fingerprint(record),
        "episode_review_status": "todo",
        "segments": nodes,
    }


def validate_curation(curation: dict, record: dict) -> None:
    if curation.get("schema_version") != CURATION_SCHEMA_VERSION:
        raise ValueError("Unsupported curation schema version")
    if curation.get("episode_index") != record.get("episode_index"):
        raise ValueError("Curation and segmentation episode IDs differ")
    if curation.get("source_repo_id") != record.get("source_repo_id"):
        raise ValueError("Curation and segmentation source datasets differ")
    if curation.get("source_revision") != record.get("source_revision"):
        raise ValueError("Curation and segmentation source revisions differ")
    if curation.get("segment_record_fingerprint") != fingerprint(record):
        raise ValueError("Curation is based on a different segmentation sidecar")
    if curation.get("n_frames") != record.get("n_frames"):
        raise ValueError("Curation and segmentation frame counts differ")
    if not float(curation.get("source_fps", 0)) > 0 or curation.get("source_fps") != record.get("fps"):
        raise ValueError("Curation and segmentation frame rates differ")
    if curation.get("bounds") != "half_open_source_frames":
        raise ValueError("Curation bounds must use half-open source-frame intervals")
    if not curation.get("segment_repo_id") or not curation.get("segment_revision"):
        raise ValueError("Curation must retain its segmentation repository and pinned revision")
    if curation.get("episode_review_status") not in {"todo", "accepted", "uncertain", "skipped"}:
        raise ValueError("Invalid episode review status")
    selected = set(record.get("selected_node_ids", []))
    segments = curation.get("segments")
    if not isinstance(segments, dict) or set(segments) != selected:
        raise ValueError("Curation must contain exactly the selected segmentation nodes")
    nodes = {node["id"]: node for node in record.get("nodes", [])}
    cursor = 0
    for node_id, segment in sorted(segments.items(), key=lambda item: item[1]["start_frame"]):
        node = nodes[node_id]
        start, end = segment.get("start_frame"), segment.get("end_frame")
        if type(start) is not int or type(end) is not int or start != cursor or not start < end <= curation["n_frames"]:
            raise ValueError(f"Segment {node_id} must form contiguous half-open episode coverage")
        if segment.get("review_status") not in {"todo", "accepted", "uncertain", "skipped"}:
            raise ValueError(f"Segment {node_id} has an invalid review status")
        if not isinstance(segment.get("text", ""), str) or not isinstance(segment.get("points"), list):
            raise ValueError(f"Segment {node_id} has invalid text or points")
        for point in segment["points"]:
            if (not isinstance(point, dict) or not isinstance(point.get("label"), str)
                    or not point["label"].strip()):
                raise ValueError(f"Segment {node_id} contains an invalid point label")
            frame = point.get("seed_frame")
            xy = point.get("xy")
            if type(frame) is not int or not start <= frame < end:
                raise ValueError(f"Point seed for {node_id} is outside its segment")
            if (not isinstance(xy, list) or len(xy) != 2
                    or any(type(value) not in (int, float) for value in xy)
                    or any(not math.isfinite(float(value)) for value in xy)):
                raise ValueError(f"Point seed for {node_id} must contain pixel x/y coordinates")
        cursor = end
    if cursor != curation["n_frames"]:
        raise ValueError("Curated segments must cover the full episode")


def remote_episode_json(repo_id: str, revision: str, path: str, cache_dir: Path) -> dict:
    from huggingface_hub import hf_hub_download

    local = Path(hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset",
                                 revision=revision, cache_dir=cache_dir))
    return json.loads(local.read_text(encoding="utf-8"))
