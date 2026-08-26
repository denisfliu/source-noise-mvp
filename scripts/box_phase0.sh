#!/usr/bin/env bash
# Phase 0 on the box: patch openpi, install snmvp, run all cheap checks.
# Confined to ~/code. Idempotent — safe to rerun. Launches NO long jobs.
set -euo pipefail

# The venv carries a cu128 torch override for Blackwell (sm_120); openpi's
# lockfile pins cu126, and a bare `uv run` re-syncs to the lock, silently
# reverting the override. Never sync here.
export UV_NO_SYNC=1

REPO="${HOME}/code/source-noise-mvp"
OPENPI="${HOME}/code/openpi"
PATCH="${REPO}/patches/openpi_arm_c_training.patch"

echo "== 1/5 openpi checkout =="
[ -d "${OPENPI}" ] || { echo "openpi missing — run scripts/setup_ec2.sh first"; exit 1; }
cd "${OPENPI}"
git rev-parse --short HEAD

echo "== 2/5 apply training patch =="
if git apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo "already applied"
elif git apply --check "${PATCH}" 2>/dev/null; then
  git apply "${PATCH}" && echo "applied"
else
  echo "PATCH DOES NOT APPLY — openpi drifted from commit 15a9616."
  echo "Options: git -C ${OPENPI} checkout 15a9616  (pin), or re-derive the"
  echo "patch per docs/openpi_integration.md (the hook is ~10 lines)."
  exit 1
fi
uv run python -m py_compile scripts/train_pytorch.py && echo "train_pytorch.py compiles"

echo "== 3/5 snmvp into openpi venv =="
uv pip install -e "${REPO}"
uv run python "${REPO}/tests/test_source_constructor.py"

echo "== 4/5 GPU / torch =="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
uv run python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
cap = torch.cuda.get_device_capability()
print("capability", cap)
if cap[0] >= 10:
    x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
    torch.cuda.synchronize()
    print("Blackwell kernel smoke test: ok")
EOF

echo "== 5/5 in-loop pin check (CPU, no data needed) =="
uv run python - <<'EOF'
import os, torch
os.environ["SNMVP_PIN_ALPHA"] = "1.0"
from snmvp import SourceConstructor, extract_invariant, carried_residual
actions = torch.randn(4, 50, 32)          # fake normalized batch
noise = torch.randn_like(actions)
noise = SourceConstructor(alpha=1.0)(noise, extract_invariant(actions))
assert carried_residual(noise, actions).abs().max() < 1e-4
for t in (0.0, 0.37, 1.0):                # pi0 interpolant convention
    x_t = t * noise + (1 - t) * actions
    assert torch.allclose(extract_invariant(x_t), extract_invariant(actions), atol=1e-4)
u_t = noise - actions
assert extract_invariant(u_t).abs().max() < 1e-4
print("carried-invariant property verified with pi0's exact conventions")
EOF

cat <<'NEXT'

All cheap checks passed. Next (GPU, launch when ready):

  # NOTE: always UV_NO_SYNC=1 (or uv run --no-sync) — a plain `uv run` reverts
  # the venv's cu128 torch override to the locked cu126 build (no sm_120).

  # A) alpha=0 parity: two short identical runs, one env var apart —
  #    loss curves must match exactly (proves the patch is inert when off)
  cd ~/code/openpi
  UV_NO_SYNC=1 uv run scripts/compute_norm_stats.py --config-name pi0_libero
  UV_NO_SYNC=1 SNMVP_PIN_ALPHA=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_pytorch.py pi0_libero --exp_name parity_off  # stop after ~50 steps (ctrl-c)
  UV_NO_SYNC=1 SNMVP_PIN_ALPHA=0 ...                     # rerun, same seed, diff the logged losses

  # B) arm C overfit probe: ~100 steps on a small subset, then wrong-invariant
  #    probe via snmvp.openpi_adapter.make_calibrated_noise + policy.infer
  UV_NO_SYNC=1 SNMVP_PIN_ALPHA=1.0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_pytorch.py pi0_libero --exp_name armC_overfit

  # C) full Phase 1 arms per docs/mvp_plan.md gates
NEXT
