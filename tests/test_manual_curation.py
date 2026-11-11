from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from robot101.data.utilities.curation import (
    CURATION_SCHEMA_VERSION, fingerprint, fresh_curation, validate_curation,
)
from robot101.data.combine_datasets import join


SOURCE_REVISION = "a" * 40


def segmentation_record():
    return {
        "schema_version": 1,
        "episode_index": 7,
        "source_repo_id": "owner/source",
        "source_revision": SOURCE_REVISION,
        "fps": 30.0,
        "n_frames": 12,
        "bounds": "half_open_source_frames",
        "task": "Move the block",
        "selected_node_ids": ["n000", "n001"],
        "nodes": [
            {"id": "n000", "start_frame": 0, "end_frame": 6,
             "caption": {"status": "pending"}},
            {"id": "n001", "start_frame": 6, "end_frame": 12,
             "caption": {"status": "pending"}},
        ],
    }


def accepted_curation(record):
    value = fresh_curation(record, "owner/annotations", "b" * 40)
    value["episode_review_status"] = "accepted"
    value["segments"]["n000"].update(
        text="Reach for the block", review_status="accepted",
        points=[{"id": "p1", "label": "block", "seed_frame": 2,
                 "xy": [12.5, 20.0], "camera": "wrist"}],
    )
    value["segments"]["n001"].update(
        text="Move the block to the tray", review_status="accepted",
        points=[{"id": "p2", "label": "tray", "seed_frame": 8,
                 "xy": [30.0, 40.0], "camera": "wrist"}],
    )
    return value


def test_curation_validates_half_open_coverage_and_manual_seeds():
    record = segmentation_record()
    curation = accepted_curation(record)
    assert curation["schema_version"] == CURATION_SCHEMA_VERSION
    validate_curation(curation, record)

    curation["segments"]["n000"]["points"][0]["seed_frame"] = 6
    with pytest.raises(ValueError, match="outside its segment"):
        validate_curation(curation, record)


def test_manual_curated_sidecars_join_for_downstream_assembly():
    record = segmentation_record()
    curation = accepted_curation(record)
    track_sidecar = {
        "episode_index": 7,
        "source_repo_id": record["source_repo_id"],
        "source_revision": SOURCE_REVISION,
        "status": "ok",
        "camera": "wrist",
        "tracks_file": "ep000007.npz",
        "curation_fingerprint": fingerprint(curation),
        "points": [{"point_id": "p1", "node_id": "n000", "label": "block",
                    "status": "ok", "coordinates_key": "xy_0000",
                    "visibility_key": "visible_0000"},
                   {"point_id": "p2", "node_id": "n001", "label": "tray",
                    "status": "ok", "coordinates_key": "xy_0001",
                    "visibility_key": "visible_0001"}],
    }
    episode = (record["source_repo_id"], SOURCE_REVISION, 7)
    indexed = {
        episode: {
            "captioning": (record, "input/captioning/episodes/ep000007.json"),
            "curation": (curation, "input/curation/episodes/ep000007.json"),
            "manual_pov": (track_sidecar, "input/pov_tracks/episodes/ep000007.json"),
        }
    }
    rows, reviews = join(indexed)
    assert reviews == []
    assert [row["caption"] for row in rows] == ["Reach for the block", "Move the block to the tray"]
    assert rows[0]["start_frame"] == 0 and rows[0]["end_frame"] == 6
    assert '"tracks_file": "input/pov_tracks/episodes/ep000007.npz"' in rows[0]["grounding_json"]


def test_boundary_edit_makes_old_manual_tracks_stale_in_bundle():
    record = segmentation_record()
    curation = accepted_curation(record)
    old_tracks = {
        "episode_index": 7,
        "source_repo_id": record["source_repo_id"],
        "source_revision": SOURCE_REVISION,
        "status": "ok",
        "camera": "wrist",
        "tracks_file": "ep000007.npz",
        "curation_fingerprint": fingerprint(curation),
        "points": [{"point_id": "p1", "node_id": "n000", "label": "block", "status": "ok"}],
    }
    curation["segments"]["n000"]["end_frame"] = 5
    curation["segments"]["n001"]["start_frame"] = 5
    validate_curation(curation, record)
    episode = (record["source_repo_id"], SOURCE_REVISION, 7)
    rows, reviews = join({episode: {
        "captioning": (record, "input/captioning/episodes/ep000007.json"),
        "curation": (curation, "input/curation/episodes/ep000007.json"),
        "manual_pov": (old_tracks, "input/pov_tracks/episodes/ep000007.json"),
    }})
    assert "manual_pov_tracks_stale_after_curation_edit" in rows[0]["review_reasons"]
    assert '"tracks_file": null' in rows[0]["grounding_json"]


def load_propagator():
    path = Path(__file__).parents[1] / "scripts" / "propagate_pov_points.py"
    spec = importlib.util.spec_from_file_location("propagate_pov_points_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_bidirectional_tracking_stitches_the_seed_once():
    module = load_propagator()

    class FakeTapir:
        def track_segment_backward(self, frames, points, label=""):
            # Backward tracker returns data in forward segment order.
            return np.array([[[0, 0], [1, 1], [2, 2]]], dtype=np.float32), np.ones((1, 3), bool)

        def track_video(self, frames, points, query_frame_index=0):
            return (np.array([[[2, 2], [3, 3], [4, 4]]], dtype=np.float32),
                    np.ones((1, 3), bool), None, None)

    tracks, visible = module._track_seed(FakeTapir(), list(range(5)), 2, np.array([2, 2]), "unit")
    assert tracks.tolist() == [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
    assert visible.tolist() == [True, True, True, True, True]
