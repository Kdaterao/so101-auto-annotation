# SO-101 Auto Annotation

Tools for turning SO-101 demonstrations into segmented, captioned, and spatially grounded training data. The package includes episode annotation and aggregation, preprocessing, gripper point labeling, perception adapters, dataset combining, and review utilities.

## Setup

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

For Ubuntu GPU annotation, use the isolated setup profile:

```bash
git clone --recurse-submodules https://github.com/Kdaterao/so101-auto-annotation.git
cd so101-auto-annotation
bash scripts/setup/linux_setup.sh --profile annotation
source .venv-annotation/bin/activate
```

Setup installs the project and checks CUDA, video decoding, and CLI imports.
Run `bash scripts/setup/linux_setup.sh --profile annotation --check` to check an
existing installation. For spatial grounding, select `--profile grounded` and
activate `.venv-grounded`. See [the single VM guide](docs/single_vm_caption_pipeline.md)
for the full caption pipeline.

For generic development and viewers:

```bash
git clone --recurse-submodules https://github.com/Kdaterao/so101-auto-annotation.git
cd so101-auto-annotation
uv sync --extra workflow --extra dev
uv pip install --no-deps -e ./tapnet
```

If you cloned without submodules, fetch the pinned upstream dependencies with `git submodule update --init --recursive` before syncing. Model weights and Hugging Face credentials are configured separately; see the workflow guides in `docs/`.

## Main workflows

- Annotate and segment episodes: `uv run python -m robot101.data.annotate --help`
- Aggregate generated captions: `uv run python -m robot101.data.aggregate_annotations --help`
- Ground and preprocess episodes: `uv run python -m robot101.data.preprocess --help`
- Combine datasets: `uv run python -m robot101.data.combine_datasets --help`
- Collect gripper point labels: `uv run python -m robot101.data.collect_points --help`
- Inspect annotation outputs: `uv run streamlit run scripts/view_episode_annotations.py`
- Manually curate published segments: `uv run python scripts/curate_episode_annotations.py --help`
- Propagate human-clicked POV points: `uv run python scripts/propagate_pov_points.py --help`

The model pipeline uses JEPA features for temporal segmentation, vision-language models for action captions, and perception/tracking models for spatial grounding. The package's workflow guides describe model setup, data formats, caching, and staged processing.

For a manual review workflow that avoids automatic POV clustering, see
[manual curation and direct POV tracking](docs/manual_curation_and_pov_tracking.md).
The curation app is separate from the read-only annotation viewer. Segmentation,
curation, viewing, and direct POV tracking all use the same `.venv`.

## Data

`data/locate_gripper/` contains the small image-labeling set used by the gripper-point workflow. Demonstration datasets and model weights are not bundled; tools download or read them from the paths configured by each command.

## Development

```bash
uv run pytest
uv run ruff check src tests scripts
```

The `lerobot` and `tapnet` directories are upstream Git submodules pinned to the revisions used by this project. Keep them as external dependencies; do not commit local edits to those checkouts here.
