#!/usr/bin/env bash
# QUEUED step-sweep on task 21 (full demos). Waits for the k-sweep to finish, then
# adapts pin and scratch saving a checkpoint every 500 steps (kept), and evals each
# step checkpoint -> success-vs-steps learning curves. GPU 0 ONLY (safe unattended;
# leaves GPU 1 free for Denis). Robust to SSH drops: self-contained, status-file driven.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
PIN=$CK/snmvp_src_pin/4999/params
SCR=$CK/snmvp_src_scratch/4999/params
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"

# wait for the k-sweep to complete (frees the GPUs)
until grep -q KSWEEP_DONE "$RD/ksweep.status" 2>/dev/null; do sleep 60; done
sleep 60
rm -f "$RD/stepsweep.status"

adapt() {  # exp init pinU|"" episodes
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    SNMVP_INIT_CKPT="$2" SNMVP_PIN_U="$3" SNMVP_EPISODES="$4" \
    $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=$1 \
    --num-train-steps=3000 --save-interval=500 --keep-period=500 --overwrite > "$RD/$1.log" 2>&1
  echo "ADAPT_$1=$?" >> "$RD/stepsweep.status"
}
adapt fs_pin_t21_steps     "$PIN" "$RD/pin_U_pca_k5.npy" "$RD/fs_t21_full.json"
adapt fs_scratch_t21_steps "$SCR" ""                     "$RD/fs_t21_full.json"
echo "ADAPTS_DONE" >> "$RD/stepsweep.status"

for S in 500 1000 1500 2000 2500 2999; do
  [ -d "$CK/fs_pin_t21_steps/$S" ] && bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21_steps/$S" \
      "$RD/prior_t21.npz" 4 8001 0 "$RD/cl_pin_t21_s$S.json" libero_object
  [ -d "$CK/fs_scratch_t21_steps/$S" ] && bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t21_steps/$S" \
      NONE 4 8001 0 "$RD/cl_scratch_t21_s$S.json" libero_object
  echo "EVAL_STEP_$S done" >> "$RD/stepsweep.status"
done
echo "STEPSWEEP_DONE" >> "$RD/stepsweep.status"
