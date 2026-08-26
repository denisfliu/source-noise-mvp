#!/usr/bin/env bash
# Scale the "both cases" pin to the VLA: retrain the source-pin FLOW on sim (LIBERO) under
# pi0_libero_shared (no-delta) pinned with the RRR subspace (action directions predictable from
# state+language jointly) -- the basis that beat PCA in every regime in the gate analysis. This
# frozen flow will be driven by a (state+language)->c prior for closed-loop tests on both a
# state-based held-out task and a language-based one.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/rrr_src.status"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  SNMVP_PIN_U="$RD/pin_U_rrr_k5_shared.npy" SNMVP_EPISODES="$RD/source_episodes.json" \
  $UV run scripts/train.py pi0_libero_shared --exp-name=snmvp_src_pin_rrr \
  --num-train-steps=5000 --save-interval=2500 --overwrite > "$RD/src_pin_rrr.log" 2>&1
echo "RRR_SRC_DONE=$? $(date -u +%H:%M:%S)" >> "$RD/rrr_src.status"
