#!/usr/bin/env bash
# Ubuntu GPU setup. Jobs and model downloads are run separately; see docs/.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PROFILE=grounded
CHECK_ONLY=0
STEP='argument validation'
trap 'status=$?; printf "\nSetup failed during %s (profile=%s, line=%s, exit=%s).\nFix the error above, then rerun the same setup command.\n" "$STEP" "$PROFILE" "$LINENO" "$status" >&2; exit "$status"' ERR

usage() {
  cat <<'HELP'
Usage: bash scripts/setup/linux_setup.sh [--profile grounded|molmo2|annotation|robot|all] [--check]

  annotation  Segmentation + PerceptionLM/Qwen captions + Claude (.venv-annotation)
  grounded    Molmo2 inference + TAPIR + SAM 2 (.venv-grounded; default)
  molmo2      Molmo2 connector training (.venv-molmo2)
  robot       LeRobot control/training (.venv)
  all         Set up and verify all four isolated environments
  --check     Verify existing environments without installing packages

SKIP_APT=1 skips Ubuntu system packages if they are already installed.
TORCH_INDEX_URL may override the default https://download.pytorch.org/whl/cu130.
Requires an NVIDIA driver compatible with the selected PyTorch CUDA wheel.
Setup does not download model checkpoints, launch jobs, or request HF/API tokens.
HELP
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      PROFILE="$2"; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$PROFILE" in
  grounded|molmo2|annotation|robot|all) ;;
  *) echo "Unknown profile: $PROFILE" >&2; usage >&2; exit 2 ;;
esac
if [[ "$(uname -s)" != Linux ]]; then
  echo 'This installer requires Linux. Use scripts/setup/windows_setup.ps1 on Windows.' >&2
  exit 2
fi
# Detect a broken requirements catalog before installing anything.
for selected in annotation grounded molmo2 robot; do
  if [[ "$PROFILE" == all || "$PROFILE" == "$selected" ]]; then
    if ! grep -Fqx "# [profile:${selected}]" requirements.txt; then
      echo "Missing requirements profile '${selected}'. Update this standalone checkout's requirements.txt and installer together." >&2
      exit 2
    fi
  fi
done

if [[ "$CHECK_ONLY" == 0 ]]; then
  STEP='NVIDIA driver check'
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 'nvidia-smi is missing. Install/enable the VM NVIDIA driver first.' >&2
    exit 1
  fi
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  STEP='Ubuntu system packages'
  if [[ "${SKIP_APT:-0}" != 1 ]]; then
    command -v apt-get >/dev/null || { echo 'Install system packages and set SKIP_APT=1 on non-Ubuntu Linux.' >&2; exit 2; }
    SUDO=()
    if [[ "$EUID" != 0 ]]; then
      command -v sudo >/dev/null || { echo 'sudo is required to install system packages; or set SKIP_APT=1.' >&2; exit 2; }
      SUDO=(sudo)
    fi
    "${SUDO[@]}" apt-get update
    # uv supplies Python 3.12, including on Ubuntu versions without python3.12 apt packages.
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
fi

prepare_submodule() {
  local name="$1"
  STEP="${name} submodule"
  git submodule update --init --recursive "$name"
  [[ -f "$ROOT/$name/pyproject.toml" || -f "$ROOT/$name/setup.py" ]] || {
    echo "Submodule $name has no Python project metadata. Check the submodule checkout." >&2; exit 1;
  }
}

prepare_environment() {
  STEP="${PROFILE} virtual environment"
  if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    uv venv --seed --python 3.12 "$ENV_DIR"
  fi
  "$ENV_DIR/bin/python" -I - "$ENV_DIR" <<'PY'
from pathlib import Path
import sys
if sys.version_info[:2] != (3, 12) or Path(sys.prefix).resolve() != Path(sys.argv[1]).resolve():
    raise SystemExit(f'Expected a Python 3.12 venv at {sys.argv[1]}; move the incompatible venv aside and rerun setup.')
(Path(sys.prefix) / '.setup-ready.json').unlink(missing_ok=True)
print('Installing into:', sys.executable, flush=True)
PY
  STEP="${PROFILE} requirements validation"
  "$ENV_DIR/bin/python" scripts/install_requirements.py --profile "$PROFILE" --print >/dev/null
  STEP='CUDA PyTorch and matching CPU video decoder'
  uv pip install --python "$ENV_DIR/bin/python" torch==2.10.0 torchvision==0.25.0 \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
  uv pip install --python "$ENV_DIR/bin/python" --no-deps torchcodec==0.10.0 \
    --index-url https://download.pytorch.org/whl/cpu
  # Execute a CUDA operation before downloading the rest of the stack.
  "$ENV_DIR/bin/python" -I - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Check the NVIDIA driver and TORCH_INDEX_URL before continuing.')
x = torch.ones(1, device='cuda') + 1
torch.cuda.synchronize()
print('CUDA operation passed:', torch.cuda.get_device_name(0), 'torch', torch.__version__, flush=True)
PY
}

install_profile() {
  STEP="${PROFILE} dependencies and editable project"
  "$ENV_DIR/bin/python" scripts/install_requirements.py --profile "$PROFILE" --installer uv \
    --pin torch==2.10.0 --pin torchvision==0.25.0 --pin torchcodec==0.10.0
}

verify_environment() {
  STEP="${PROFILE} environment verification"
  [[ -x "$ENV_DIR/bin/python" ]] || { echo "Missing $ENV_DIR; run setup without --check first." >&2; exit 1; }
  # -I ignores PYTHONPATH and the script directory: the installed package must work.
  "$ENV_DIR/bin/python" -I "$ROOT/scripts/verify_setup.py" --profile "$PROFILE"
  printf '\nSetup verified. Activate with:\nsource %s/bin/activate\n' "$ENV_DIR"
}

setup_profile() {
  PROFILE="$1"
  case "$PROFILE" in
    robot) ENV_DIR="$ROOT/.venv" ;;
    *) ENV_DIR="$ROOT/.venv-${PROFILE}" ;;
  esac
  if [[ "$CHECK_ONLY" == 1 ]]; then
    verify_environment
    return
  fi
  case "$PROFILE" in
    annotation|grounded|robot) prepare_submodule lerobot ;;
  esac
  if [[ "$PROFILE" == grounded ]]; then prepare_submodule tapnet; fi
  prepare_environment
  if [[ "$PROFILE" == molmo2 ]]; then
    STEP='Molmo2 training checkout'
    local work="${SO101_MOLMO2_WORKDIR:-${ROOT}/molmo2_so101_run}"
    local repo="${MOLMO2_REPO_ROOT:-${work}/molmo2}"
    local ref="${SO101_MOLMO2_REF:-f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a}"
    if [[ ! -f "$repo/launch_scripts/sft.py" ]]; then
      mkdir -p "$(dirname "$repo")"
      git clone --depth 1 https://github.com/allenai/molmo2.git "$repo"
      git -C "$repo" fetch --depth 1 origin "$ref"
      git -C "$repo" checkout --detach FETCH_HEAD
    fi
    STEP='Molmo2 training dependencies'
    uv pip install --python "$ENV_DIR/bin/python" -e "$repo[train]" \
      torch==2.10.0 torchvision==0.25.0 torchcodec==0.10.0
  fi
  SAM2_BUILD_CUDA=0 install_profile
  if [[ "$PROFILE" == grounded ]]; then
    STEP='TAPIR package and spaCy language model'
    uv pip install --python "$ENV_DIR/bin/python" --no-deps -e "$ROOT/tapnet"
    "$ENV_DIR/bin/python" -m spacy download en_core_web_sm
  fi
  verify_environment
}

if [[ "$PROFILE" == all ]]; then
  for selected in grounded molmo2 robot annotation; do setup_profile "$selected"; done
else
  setup_profile "$PROFILE"
fi
