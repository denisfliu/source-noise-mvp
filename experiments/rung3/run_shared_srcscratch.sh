#!/usr/bin/env bash
# Scratch source (no pin) in the no-delta shared space, the fair no-pin control for the adapted
# head-to-head: same source-pretrain regime as snmvp_src_pin_rrr but WITHOUT the pin. Both arms then
# few-shot adapt on the target task; the only difference is the pin.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/scratch_src.status"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  SNMVP_EPISODES="$RD/source_episodes.json" \
  $UV run scripts/train.py pi0_libero_shared --exp-name=snmvp_src_scratch_shared \
  --num-train-steps=5000 --save-interval=2500 --overwrite > "$RD/src_scratch_shared.log" 2>&1
echo "SCRATCH_SRC_DONE=$? $(date -u +%H:%M:%S)" >> "$RD/scratch_src.status"
