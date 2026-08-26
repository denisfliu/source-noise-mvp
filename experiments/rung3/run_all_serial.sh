#!/usr/bin/env bash
# Serial pipeline on GPU 0 only (one GPU at a time): 4 few-shot adaptations, then 4
# closed-loop evals, on the easy held-out libero_object tasks 21 (object task 4) and
# 28 (object task 5), pin+prior vs scratch. Everything runs one-at-a-time.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
PIN=$CK/snmvp_src_pin/4999/params
SCR=$CK/snmvp_src_scratch/4999/params
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/serial.status"

adapt() {  # exp init pinU|"" episodes
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    SNMVP_INIT_CKPT="$2" SNMVP_PIN_U="$3" SNMVP_EPISODES="$4" \
    $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=$1 \
    --num-train-steps=800 --save-interval=800 --overwrite > "$RD/$1.log" 2>&1
  echo "ADAPT_$1=$?" >> "$RD/serial.status"
}

adapt fs_pin_t21     "$PIN" "$RD/pin_U_pca_k5.npy" "$RD/fs_t21_k10.json"
adapt fs_scratch_t21 "$SCR" ""                     "$RD/fs_t21_k10.json"
adapt fs_pin_t28     "$PIN" "$RD/pin_U_pca_k5.npy" "$RD/fs_t28_k10.json"
adapt fs_scratch_t28 "$SCR" ""                     "$RD/fs_t28_k10.json"
echo "ADAPTS_DONE" >> "$RD/serial.status"

bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21/799"     "$RD/prior_t21.npz" 4 8001 0 "$RD/cl_pin_t21.json"     libero_object
bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t21/799" NONE               4 8001 0 "$RD/cl_scratch_t21.json" libero_object
bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t28/799"     "$RD/prior_t28.npz" 5 8002 0 "$RD/cl_pin_t28.json"     libero_object
bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t28/799" NONE               5 8002 0 "$RD/cl_scratch_t28.json" libero_object
echo "ALL_SERIAL_DONE" >> "$RD/serial.status"
