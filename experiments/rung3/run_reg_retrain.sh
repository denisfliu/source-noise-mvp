#!/usr/bin/env bash
# Address the pin's late-step decline with regularization (the config had ~zero weight
# decay and EMA off). Retrain the pin adapter on the same 45 demos and seed but with
# weight_decay=1e-3 and EMA=0.99, step-checkpointed, then re-eval the curve. Isolates
# the regularization effect (same data/seed). GPU 0 only.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
PIN=$CK/snmvp_src_pin/4999/params
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/regfix.status"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  SNMVP_INIT_CKPT="$PIN" SNMVP_PIN_U="$RD/pin_U_pca_k5.npy" SNMVP_EPISODES="$RD/fs_t21_full.json" \
  SNMVP_WD=1e-3 SNMVP_EMA=0.99 \
  $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=fs_pin_t21_reg \
  --num-train-steps=3000 --save-interval=500 --keep-period=500 --overwrite > "$RD/fs_pin_t21_reg.log" 2>&1
echo "REG_ADAPT_DONE=$?" >> "$RD/regfix.status"

for S in 500 1000 1500 2000 2500 2999; do
  [ -d "$CK/fs_pin_t21_reg/$S" ] && bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21_reg/$S" \
    "$RD/prior_t21.npz" 4 8001 0 "$RD/cl_pin_t21_reg_s${S}.json" libero_object 30
  echo "REG_STEP_$S done" >> "$RD/regfix.status"
done
echo "REGFIX_DONE" >> "$RD/regfix.status"
