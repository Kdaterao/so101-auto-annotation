# Manual curation and direct POV point tracking

This workflow is separate from `scripts/view_episode_annotations.py`, which stays
read-only. It also bypasses `robot101.data.preprocess`: that command performs POV
candidate clustering and Molmo/SAM grounding. This workflow uses human point clicks
as the only TAPIR seeds.

## 1. Generate and publish segment sidecars

Run the existing JEPA segment stage without generating captions. Sidecars are
published under `captioning/episodes/` with the source episode/revision, frame
timestamps, full hierarchy, and selected subtask cut.

```bash
uv run python -m robot101.data.annotate \
  --stage segment \
  --episodes 0-9 \
  --output-dir outputs/segments_test \
  --push-to-hub \
  --dst-repo-id YOUR_USER/so101_manual_curation
```

Use `--resume` to continue or publish another disjoint shard to the same dataset.
The segment command pins the source dataset revision in its sidecars. The UI loads
the source videos from that exact revision.

## 2. Review and publish human annotations

Install the optional local transcription dependency once:

```bash
uv sync --extra workflow
uv pip install --no-deps -e ./tapnet
```

Start the separate curation app:

```bash
uv run python scripts/curate_episode_annotations.py \
  --repo-id YOUR_USER/so101_manual_curation \
  --revision main \
  --output-dir outputs/manual_curation \
  --push-to-hub \
  --open-browser
```

The app pins the segment repository revision at startup. Dictation uses the local
`faster-whisper` model (default `base.en`, CPU int8); recordings are kept in memory
only while being transcribed and are deleted immediately afterward. Set
`--stt-model`, `--stt-device`, and `--stt-compute-type` to choose a different local
model/runtime.

Drafts auto-save under `outputs/manual_curation/curation/episodes/`. Use **Publish
episode to Hugging Face** to upload that episode under `curation/episodes/` in the
same dataset as the segment sidecars. An accepted episode still requires each
tracked segment to be marked accepted. Seeds are labeled click coordinates in
native wrist-camera pixels and source-frame indices. Moving a shared segment
boundary updates both adjacent intervals and removes seeds that no longer fall
inside their interval.

## 3. Propagate accepted POV clicks

Use the shared `.venv` and download the checkpoint described in
[grounded training data](grounded_training_data.md), then run the standalone
propagator:

```bash
uv run python scripts/propagate_pov_points.py \
  --repo-id YOUR_USER/so101_manual_curation \
  --revision main \
  --episodes 0-9 \
  --output-dir outputs/pov_tracks \
  --device cuda \
  --resume \
  --push-to-hub
```

The command pins the curation dataset revision, then retrieves the separately
pinned segment and source revisions from each curation record. For each accepted
seed it tracks backward from the click frame to the segment start and forward to
the segment end, stitches the tracks without duplicating the seed frame, and
preserves the clicked pixel exactly at that frame. It masks visibility where the
source POV camera frame is missing.

The script writes a compact compressed `epNNNNNN.npz` with coordinate and
visibility arrays, plus a JSON record with one entry per seed and references to
those arrays. Both files are uploaded under `pov_tracks/episodes/`. A failed
track stays flagged for review; it is not replaced by a guessed point. Resume
skips outputs only when the curation, checkpoint, camera and source fingerprint
match.

## Downstream assembly

`robot101.data.combine_datasets` collects `curation/episodes/` and
`pov_tracks/episodes/` alongside captioning and aggregation records. Manual text
becomes the subtask caption/action text, edited frame bounds become the curated
cut, and point-track metadata in `grounding_json` references the copied NPZ file.
The combined output preserves the complete input sidecars under `artifacts/`.
Unaccepted annotations, missing text/points, and failed tracks appear in
`review.json`; the dataset does not silently treat them as verified labels.

```bash
uv run python -m robot101.data.combine_datasets \
  --repos YOUR_USER/so101_manual_curation \
  --output-dir outputs/manual_curation_bundle
```

The JSON sidecars identify the source revision and use half-open source-frame
bounds `[start_frame, end_frame)`. The NPZ keys referenced by each point entry
contain `[frames, 2]` pixel coordinates and a `[frames]` visibility mask for that
point's segment interval.
