#!/usr/bin/env bash
# Decisive control for the pin's late-step decline: an INDEPENDENT training seed.
# The prior-refit and wd+ema fixes both failed, but they shared the original seed. This
# retrains the pin adapter identically (same 45 demos, init ckpt, pin U, 3000 steps, no
# extra reg) but with a different seed, step-checkpointed, then re-evals the curve. If it
# still peaks-then-declines -> the decline is a real, seed-independent property of the
# pin+flow interaction under prolonged training (not single-run luck). GPU 0 only.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
PIN=$CK/snmvp_src_pin/4999/params
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/seed2.status"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  SNMVP_INIT_CKPT="$PIN" SNMVP_PIN_U="$RD/pin_U_pca_k5.npy" SNMVP_EPISODES="$RD/fs_t21_full.json" \
  $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=fs_pin_t21_seed2 \
  --seed=1 --num-train-steps=3000 --save-interval=500 --keep-period=500 --overwrite > "$RD/fs_pin_t21_seed2.log" 2>&1
echo "SEED2_ADAPT_DONE=$?" >> "$RD/seed2.status"

for S in 500 1000 1500 2000 2500 2999; do
  [ -d "$CK/fs_pin_t21_seed2/$S" ] && bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21_seed2/$S" \
    "$RD/prior_t21.npz" 4 8001 0 "$RD/cl_pin_t21_seed2_s${S}.json" libero_object 30
  echo "SEED2_STEP_$S done" >> "$RD/seed2.status"
done
echo "SEED2_DONE" >> "$RD/seed2.status"
