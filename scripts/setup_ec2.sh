#!/usr/bin/env bash
# Setup confined to ~/code. No system packages, no global installs, nothing
# outside this directory tree is touched (existing work on the instance is
# left alone).
set -euo pipefail

CODE_DIR="${HOME}/code"
REPO_DIR="${CODE_DIR}/source-noise-mvp"
OPENPI_DIR="${CODE_DIR}/openpi"

mkdir -p "${CODE_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install (user-local, no sudo):"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [ ! -d "${OPENPI_DIR}" ]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "${OPENPI_DIR}"
else
  echo "openpi already present at ${OPENPI_DIR}, leaving as is"
fi

cd "${OPENPI_DIR}"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install -e "${REPO_DIR}"

echo
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
uv run python -c "import torch; print('torch', torch.__version__, '| capability', torch.cuda.get_device_capability() if torch.cuda.is_available() else 'no cuda')" || \
  echo "torch check failed — see docs/openpi_integration.md Blackwell notes (likely need cu128+ build)"

echo
echo "Running snmvp unit tests in the training venv:"
uv run python "${REPO_DIR}/tests/test_source_constructor.py"

echo
echo "Done. Next: docs/openpi_integration.md sanity sequence."
