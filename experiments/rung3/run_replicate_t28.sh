#!/usr/bin/env bash
# Replicate the positive full-demo result on held-out task 28 (libero_object task 5),
# parallel across both GPUs: pin on GPU 0, scratch on GPU 1; then evals in parallel.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
PIN=$CK/snmvp_src_pin/4999/params
SCR=$CK/snmvp_src_scratch/4999/params
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/rep28.status"

adapt() {  # gpu exp init pinU|"" episodes
  CUDA_VISIBLE_DEVICES=$1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    SNMVP_INIT_CKPT="$3" SNMVP_PIN_U="$4" SNMVP_EPISODES="$5" \
    $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=$2 \
    --num-train-steps=3000 --save-interval=3000 --overwrite > "$RD/$2.log" 2>&1
}
( adapt 0 fs_pin_t28_full     "$PIN" "$RD/pin_U_pca_k5.npy" "$RD/fs_t28_full.json" ) &
( adapt 1 fs_scratch_t28_full "$SCR" ""                     "$RD/fs_t28_full.json" ) &
wait
echo "ADAPTS_DONE" >> "$RD/rep28.status"

( bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t28_full/2999"     "$RD/prior_t28.npz" 5 8007 0 "$RD/cl_pin_t28_full.json"     libero_object ) &
( bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t28_full/2999" NONE               5 8008 1 "$RD/cl_scratch_t28_full.json" libero_object ) &
wait
echo "REP28_DONE" >> "$RD/rep28.status"
