#!/usr/bin/env bash
# Ubuntu GPU setup. Jobs and model downloads are run separately; see docs/.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PROFILE=workflow
STEP='argument validation'
trap 'status=$?; printf "\nSetup failed during %s (profile=%s, line=%s, exit=%s).\nFix the error above, then rerun the same setup command.\n" "$STEP" "$PROFILE" "$LINENO" "$status" >&2; exit "$status"' ERR

usage() {
  cat <<'HELP'
Usage: bash scripts/setup/linux_setup.sh [--profile workflow|molmo2]

  workflow  JEPA segmentation, curation UI, POV BootsTAPIR, and grounding (.venv; default)
  molmo2    Optional Molmo2 connector training dependencies (.venv-molmo2)

SKIP_APT=1 skips Ubuntu system packages if they are already installed.
TORCH_INDEX_URL may override the default CUDA PyTorch wheel index.
Requires an NVIDIA driver compatible with the selected CUDA wheel.
Setup does not download model checkpoints, launch jobs, or request HF/API tokens.
HELP
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      PROFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$PROFILE" in
  workflow|molmo2) ;;
  *) echo "Unknown profile: $PROFILE" >&2; usage >&2; exit 2 ;;
esac
if [[ "$(uname -s)" != Linux ]]; then
  echo 'This installer requires Linux. Use scripts/setup/windows_setup.ps1 on Windows.' >&2
  exit 2
fi

STEP='NVIDIA driver check'
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'nvidia-smi is missing. Install/enable the VM NVIDIA driver first.' >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
STEP='Ubuntu system packages'
if [[ "${SKIP_APT:-0}" != 1 ]]; then
  command -v apt-get >/dev/null || { echo 'Install system dependencies and set SKIP_APT=1 on non-Ubuntu Linux.' >&2; exit 2; }
  SUDO=()
  if [[ "$EUID" != 0 ]]; then
    command -v sudo >/dev/null || { echo 'sudo is required to install system packages; or set SKIP_APT=1.' >&2; exit 2; }
    SUDO=(sudo)
  fi
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y git curl ca-certificates ffmpeg tmux build-essential pkg-config
fi
for executable in git curl ffmpeg; do
  command -v "$executable" >/dev/null || { echo "Missing $executable; install system packages first." >&2; exit 2; }
done
STEP='uv installation'
if ! command -v uv >/dev/null 2>&1; then
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! command -v uv >/dev/null 2>&1; then
    curl --fail --location --silent --show-error https://astral.sh/uv/install.sh | sh
  fi
fi
command -v uv >/dev/null

prepare_submodule() {
  local name="$1"
  STEP="${name} submodule"
  git submodule update --init --recursive "$name"
  [[ -f "$ROOT/$name/pyproject.toml" || -f "$ROOT/$name/setup.py" ]] || {
    echo "Submodule $name has no Python project metadata. Check the submodule checkout." >&2; exit 1;
  }
}

prepare_environment() {
  local env_dir="$1"
  STEP='Python 3.12 virtual environment'
  if [[ ! -x "$env_dir/bin/python" ]]; then
    uv venv --seed --python 3.12 "$env_dir"
  fi
  "$env_dir/bin/python" -I - "$env_dir" <<'PY'
from pathlib import Path
import sys
if sys.version_info[:2] != (3, 12) or Path(sys.prefix).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit(f'Expected a Python 3.12 venv at {sys.argv[1]}; move the incompatible venv aside and rerun setup.')
print('Installing into:', sys.executable, flush=True)
PY
}

setup_workflow() {
  local env_dir="$ROOT/.venv"
  prepare_submodule lerobot
  prepare_submodule tapnet
  prepare_environment "$env_dir"
  STEP='shared workflow dependencies'
  uv pip install --python "$env_dir/bin/python" -e ".[workflow,dev]"
  STEP='CUDA PyTorch and matching CPU video decoder'
  uv pip install --python "$env_dir/bin/python" --reinstall torch==2.10.0 torchvision==0.25.0 \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
  uv pip install --python "$env_dir/bin/python" --no-deps torchcodec==0.10.0 \
    --index-url https://download.pytorch.org/whl/cpu
  "$env_dir/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Check the NVIDIA driver and TORCH_INDEX_URL before continuing.')
x = torch.ones(1, device='cuda') + 1
torch.cuda.synchronize()
print('CUDA operation passed:', torch.cuda.get_device_name(0), 'torch', torch.__version__, flush=True)
PY
  STEP='TAPIR, SAM 2, and spaCy workflow dependencies'
  uv pip install --python "$env_dir/bin/python" --no-deps -e ./tapnet
  SAM2_BUILD_CUDA=0 uv pip install --python "$env_dir/bin/python" \
    "sam2 @ git+https://github.com/facebookresearch/sam2.git"
  "$env_dir/bin/python" -m spacy download en_core_web_sm
  export PYTHONPATH="$ROOT/src:$ROOT/lerobot/src:${PYTHONPATH:-}"
  "$env_dir/bin/python" - <<'PY'
import spacy
import torch
import einshape
import einops
from faster_whisper import WhisperModel
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tapnet.torch import tapir_model
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from robot101.data.utilities.episode_helpers import _load_lerobot
_load_lerobot()
spacy.load('en_core_web_sm')
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: check the VM NVIDIA driver.')
print('Shared workflow environment ready on', torch.cuda.get_device_name(0))
PY
  printf '\nSetup verified. Activate with:\nsource %s/bin/activate\n' "$env_dir"
  echo 'Run commands: docs/episode_annotation.md, docs/manual_curation_and_pov_tracking.md'
}

setup_molmo2() {
  local env_dir="$ROOT/.venv-molmo2"
  prepare_environment "$env_dir"
  local work="${SO101_MOLMO2_WORKDIR:-${ROOT}/molmo2_so101_run}"
  local repo="${MOLMO2_REPO_ROOT:-${work}/molmo2}"
  local ref="${SO101_MOLMO2_REF:-f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a}"
  STEP='Molmo2 training checkout'
  if [[ ! -f "$repo/launch_scripts/sft.py" ]]; then
    mkdir -p "$(dirname "$repo")"
    git clone --depth 1 https://github.com/allenai/molmo2.git "$repo"
    git -C "$repo" fetch --depth 1 origin "$ref"
    git -C "$repo" checkout --detach FETCH_HEAD
  fi
  STEP='Molmo2 CUDA training dependencies'
  uv pip install --python "$env_dir/bin/python" torch==2.10.0 torchvision==0.25.0 \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
  uv pip install --python "$env_dir/bin/python" --reinstall --no-deps torchcodec==0.12.0 \
    --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python "$env_dir/bin/python" -e "$repo[train]"
  "$env_dir/bin/python" scripts/install_requirements.py --profile molmo2 --installer uv
  printf '\nMolmo2 setup complete. Activate with:\nsource %s/bin/activate\n' "$env_dir"
  echo 'Run commands: docs/molmo2_training.md'
}

case "$PROFILE" in
  workflow) setup_workflow ;;
  molmo2) setup_molmo2 ;;
esac
