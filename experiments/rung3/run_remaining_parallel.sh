#!/usr/bin/env bash
# Finish the easy-task cut across BOTH GPUs. fs_pin_t21 is already done.
# Phase 1 adapts: GPU0 = fs_scratch_t21 then fs_scratch_t28; GPU1 = fs_pin_t28.
# Phase 2 evals (libero_object): GPU0 = t21 (pin,scratch); GPU1 = t28 (pin,scratch).
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
PIN=$CK/snmvp_src_pin/4999/params
SCR=$CK/snmvp_src_scratch/4999/params
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/par.status"

adapt() {  # gpu exp init pinU|"" episodes
  CUDA_VISIBLE_DEVICES=$1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    SNMVP_INIT_CKPT="$3" SNMVP_PIN_U="$4" SNMVP_EPISODES="$5" \
    $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=$2 \
    --num-train-steps=800 --save-interval=800 --overwrite > "$RD/$2.log" 2>&1
}

( adapt 0 fs_scratch_t21 "$SCR" "" "$RD/fs_t21_k10.json"
  adapt 0 fs_scratch_t28 "$SCR" "" "$RD/fs_t28_k10.json" ) &
( adapt 1 fs_pin_t28 "$PIN" "$RD/pin_U_pca_k5.npy" "$RD/fs_t28_k10.json" ) &
wait
echo "ADAPTS_DONE" >> "$RD/par.status"

( bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21/799"     "$RD/prior_t21.npz" 4 8001 0 "$RD/cl_pin_t21.json"     libero_object
  bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t21/799" NONE               4 8001 0 "$RD/cl_scratch_t21.json" libero_object ) &
( bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t28/799"     "$RD/prior_t28.npz" 5 8002 1 "$RD/cl_pin_t28.json"     libero_object
  bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t28/799" NONE               5 8002 1 "$RD/cl_scratch_t28.json" libero_object ) &
wait
echo "PAR_ALL_DONE" >> "$RD/par.status"
