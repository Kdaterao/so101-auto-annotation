# SO-101 Auto Annotation

Tools for turning SO-101 demonstrations into segmented, captioned, and spatially grounded training data. The package includes episode annotation and aggregation, preprocessing, gripper point labeling, perception adapters, dataset combining, and review utilities.

## Setup

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/Kdaterao/so101-auto-annotation.git
cd so101-auto-annotation
uv sync --extra dev --extra viewer
```

If you cloned without submodules, fetch the pinned upstream dependencies with `git submodule update --init --recursive` before syncing. Model weights and Hugging Face credentials are configured separately; see the workflow guides in `docs/`.

## Main workflows

- Annotate and segment episodes: `uv run python -m robot101.data.annotate --help`
- Aggregate generated captions: `uv run python -m robot101.data.aggregate_annotations --help`
- Ground and preprocess episodes: `uv run python -m robot101.data.preprocess --help`
- Combine datasets: `uv run python -m robot101.data.combine_datasets --help`
- Collect gripper point labels: `uv run python -m robot101.data.collect_points --help`
- Inspect annotation outputs: `uv run streamlit run scripts/view_episode_annotations.py`

The model pipeline uses JEPA features for temporal segmentation, vision-language models for action captions, and perception/tracking models for spatial grounding. The package's workflow guides describe model setup, data formats, caching, and staged processing.

## Data

`data/locate_gripper/` contains the small image-labeling set used by the gripper-point workflow. Demonstration datasets and model weights are not bundled; tools download or read them from the paths configured by each command.

## Development

```bash
uv run pytest
uv run ruff check src tests scripts
```

The `lerobot` and `tapnet` directories are upstream Git submodules pinned to the revisions used by this project. Keep them as external dependencies; do not commit local edits to those checkouts here.
