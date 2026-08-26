#!/usr/bin/env bash
# Retrain the source-pin FLOW on sim (LIBERO) in the shared standardized space:
# config pi0_libero_shared (extra_delta_transform OFF, LIBERO-own raw-delta norm), pinned with
# the shared-space K=5 U. This is the frozen sim flow whose realization we test on real Bridge
# (with only the pin prior refit on real). Same source split (32 tasks) as the original run.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/shared_src.status"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  SNMVP_PIN_U="$RD/pin_U_pca_k5_shared.npy" SNMVP_EPISODES="$RD/source_episodes.json" \
  $UV run scripts/train.py pi0_libero_shared --exp-name=snmvp_src_pin_shared \
  --num-train-steps=5000 --save-interval=2500 --overwrite > "$RD/src_pin_shared.log" 2>&1
echo "SRC_SHARED_DONE=$? $(date -u +%H:%M:%S)" >> "$RD/shared_src.status"
