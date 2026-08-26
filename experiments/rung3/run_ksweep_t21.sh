#!/usr/bin/env bash
# Data-efficiency k-sweep on held-out task 21 (libero_object task 4): k=25 and k=15,
# pin+prior vs scratch, 3000-step adapts. (k=45 -> pin .80/scratch .30; k=10 -> both 0.)
# Both GPUs: GPU 0 = pin adapts, GPU 1 = scratch adapts; then evals in parallel.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
PIN=$CK/snmvp_src_pin/4999/params
SCR=$CK/snmvp_src_scratch/4999/params
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/ksweep.status"

adapt() {  # gpu exp init pinU|"" episodes
  CUDA_VISIBLE_DEVICES=$1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    SNMVP_INIT_CKPT="$3" SNMVP_PIN_U="$4" SNMVP_EPISODES="$5" \
    $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=$2 \
    --num-train-steps=3000 --save-interval=3000 --overwrite > "$RD/$2.log" 2>&1
}
( adapt 0 fs_pin_t21_k25 "$PIN" "$RD/pin_U_pca_k5.npy" "$RD/fs_t21_k25.json"
  adapt 0 fs_pin_t21_k15 "$PIN" "$RD/pin_U_pca_k5.npy" "$RD/fs_t21_k15.json" ) &
( adapt 1 fs_scratch_t21_k25 "$SCR" "" "$RD/fs_t21_k25.json"
  adapt 1 fs_scratch_t21_k15 "$SCR" "" "$RD/fs_t21_k15.json" ) &
wait
echo "ADAPTS_DONE" >> "$RD/ksweep.status"

( bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21_k25/2999" "$RD/prior_t21_k25.npz" 4 8001 0 "$RD/cl_pin_t21_k25.json" libero_object
  bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21_k15/2999" "$RD/prior_t21_k15.npz" 4 8001 0 "$RD/cl_pin_t21_k15.json" libero_object ) &
( bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t21_k25/2999" NONE 4 8002 1 "$RD/cl_scratch_t21_k25.json" libero_object
  bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t21_k15/2999" NONE 4 8002 1 "$RD/cl_scratch_t21_k15.json" libero_object ) &
wait
echo "KSWEEP_DONE" >> "$RD/ksweep.status"
