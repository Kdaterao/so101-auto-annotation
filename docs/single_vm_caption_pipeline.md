# Run the caption pipeline on one VM

This guide runs the full caption workflow on one Ubuntu GPU VM: V-JEPA 2.1
segmentation, Perception-LM-3B motion captions, Qwen3-VL-8B static image
descriptions, and Claude Haiku aggregation. Grounding is a separate workflow.

The Hugging Face dataset can hold both outputs as separate configurations:
`captioning/` contains the caption tree, and `aggregation/` contains Claude's
standardized annotations.

## 1. Set up the VM

From the repository root:

```bash
git submodule update --init lerobot
bash scripts/setup/linux_setup.sh --profile annotation
source .venv-annotation/bin/activate
hf auth login
```

Setup uses this standalone repository's `src/robot101` package and installs it
into `.venv-annotation`. It verifies imports outside the checkout, a CUDA tensor
operation, both annotation CLIs, and a small PyAV/TorchCodec video round trip.
It prints **Setup verified** only after these checks pass. Re-running the same
setup command repairs an existing environment; activating a venv by itself does
not install the project. You can rerun only the checks with:

```bash
bash scripts/setup/linux_setup.sh --profile annotation --check
```

Python 3.12 is provided by `uv`; it does not need a matching Ubuntu apt package.
Use `SKIP_APT=1 bash scripts/setup/linux_setup.sh --profile annotation` only when
the system packages are already installed. If setup fails, it names the failing
step; fix that error and rerun before starting annotation.

Use an account with access to the Perception-LM-3B model. If the run will publish
artifacts, the Hugging Face account also needs write access to the destination
dataset. Start a persistent terminal before a long job if you may disconnect:

```bash
tmux new -s captions
cd ~/so101-auto-annotation
source .venv-annotation/bin/activate
```

Set the run values in that terminal. The defaults below use the project's
default source dataset and a destination dataset you own:

```bash
export SOURCE_REPO="felsager/community_dataset_v3_ee_smolVLA"
export ANNOTATION_REPO="YOUR_HF_USERNAME/so101_caption_pipeline"
```

Replace `YOUR_HF_USERNAME` before running the commands.

## 2. Choose episodes

Episode selections are inclusive. For a quick three-episode smoke run:

```bash
export EPISODES="0-2"
```

For a larger run, set the range you want, for example:

```bash
export EPISODES="0-999"
```

To process every episode in the source dataset, query its metadata and form the
inclusive range automatically:

```bash
TOTAL_EPISODES=$(python - <<'PY'
import os
from pathlib import Path
from robot101.data.utilities.episode_helpers import DEFAULT_SRC, _load_lerobot

_, Metadata, _, cache_root = _load_lerobot()
repo_id = os.environ.get("SOURCE_REPO", DEFAULT_SRC)
metadata = Metadata(repo_id, root=Path(cache_root) / repo_id)
print(metadata.total_episodes)
PY
)
export EPISODES="0-$((TOTAL_EPISODES - 1))"
echo "Selected all $TOTAL_EPISODES episodes: $EPISODES"
```

If using a non-default source, update both `SOURCE_REPO` and the metadata snippet
to that source ID.

## 3. Run segmentation and captioning

First run the smoke selection locally, without uploading:

```bash
export OUTPUT_DIR="outputs/captions_smoke"
python -m robot101.data.annotate \
  --src-repo-id "$SOURCE_REPO" \
  --episodes "$EPISODES" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda
```

Defaults are V-JEPA 2.1 ViT-L for temporal segmentation, Perception-LM-3B for
video motion/action captions, and Qwen3-VL-8B-Instruct for static image
descriptions. The primary camera is `top`; the wrist camera supplies sparse
helper snapshots. Captioning uses batch size 1 by default to limit VRAM use.
Increase `--video-batch-size` and `--image-batch-size` only after a smoke run
shows that the VM has VRAM headroom.

Each episode report is saved under `$OUTPUT_DIR/reports/`. To inspect local
results in a browser, run the viewer:

```bash
python scripts/view_episode_annotations.py \
  --root "$OUTPUT_DIR" --max-episodes 3 --open-browser
```

For a headless VM, omit `--open-browser` and use the SSH tunnel described in the
final viewing section.

After checking the smoke captions and segmentation, change the episode range
and use a fresh output directory for the full run. This example uses the first
1,000 episodes; replace it with your intended selection:

```bash
export EPISODES="0-999"
export OUTPUT_DIR="outputs/captions_single_vm"
python -m robot101.data.annotate \
  --src-repo-id "$SOURCE_REPO" \
  --episodes "$EPISODES" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --push-to-hub \
  --dst-repo-id "$ANNOTATION_REPO"
```

If the full run is interrupted, repeat it with `--resume`, keeping the same
source, episode selection, model, prompt and batch settings. Failed captions
can be retried by resuming. Prompt/model changes need a new output directory.

## 4. Run Claude aggregation

Aggregation reads caption trees from the Hub and writes standardized annotations
into the same dataset's `aggregation/` configuration. It uses the API, not the
VM's GPU. Set your Claude API key in the current shell:

```bash
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
```

For workspace-scoped API keys, `ANTHROPIC_WORKSPACE_ID` is optional. Run a
text-only dry run first to validate the selected nodes and prompt evidence:

```bash
python -m robot101.data.aggregate_annotations \
  --src-repo-id "$ANNOTATION_REPO" \
  --episodes "$EPISODES" \
  --output-dir outputs/aggregation_single_vm \
  --node-scope selected \
  --dry-run
```

Then remove `--dry-run` and upload the real results:

```bash
python -m robot101.data.aggregate_annotations \
  --src-repo-id "$ANNOTATION_REPO" \
  --episodes "$EPISODES" \
  --output-dir outputs/aggregation_single_vm \
  --model claude-haiku-5-5 \
  --rounds 3 --concurrency 2 --max-tokens 2048 \
  --node-scope selected \
  --push-to-hub \
  --dst-repo-id "$ANNOTATION_REPO"
```

The default `selected` scope sends only selected subtasks for standardization;
their caption-tree context is included without separate Claude requests. Three
rounds means an initial annotation and two critique/revision passes. Add
`--no-review` for one round. Use `--use-task-prompt` only if you want Claude to
receive the original dataset instruction as intended-goal context.

For an interrupted aggregation run, repeat its command with `--resume`, keeping
the output directory, model, round count and token limit compatible. Lower
`--concurrency` if Claude rate limits the requests.

## 5. View the final results

View caption trees and sampled images:

```bash
python scripts/view_episode_annotations.py \
  --repo-id "$ANNOTATION_REPO" --stage captioning \
  --episodes 0-2 --open-browser
```

View the final Claude annotations:

```bash
python scripts/view_episode_annotations.py \
  --repo-id "$ANNOTATION_REPO" --stage aggregation \
  --episodes 0-2 --open-browser
```

On a headless VM, omit `--open-browser` from the viewer command and leave the
viewer running in one VM terminal. Forward its port from a separate laptop
terminal:

```bash
ssh -L 8767:127.0.0.1:8767 ubuntu@YOUR_VM_HOST
```

Open `http://127.0.0.1:8767` on the laptop. Keep the viewer process running in
the VM terminal.

## What the outputs mean

- V-JEPA 2.1 supplies temporal features for segmentation; it does not generate
  text captions.
- Perception-LM captions motion/actions from video; Qwen captions static visual
  detail from images.
- Claude combines those caption roles into structured subtask annotations. It
  receives text evidence only and does not independently re-check video pixels.
- The `completion_condition` is a desired result, not a claim that the episode
  achieved it.
- These commands do not create grounding points or modify source videos.
