#!/usr/bin/env bash
# Finish task-28 replication on GPU 0 ONLY (GPU 1 returned to Denis). Waits for the pin
# adaptation (already running on GPU 0), then runs the scratch adaptation and both evals.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
SCR=$CK/snmvp_src_scratch/4999/params
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/rep28b.status"

until [ -d "$CK/fs_pin_t28_full/2999" ]; do sleep 30; done
sleep 20   # let the pin adapt release GPU 0

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  SNMVP_INIT_CKPT="$SCR" SNMVP_PIN_U="" SNMVP_EPISODES="$RD/fs_t28_full.json" \
  $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=fs_scratch_t28_full \
  --num-train-steps=3000 --save-interval=3000 --overwrite > "$RD/fs_scratch_t28_full.log" 2>&1
echo "SCRATCH_ADAPT_DONE" >> "$RD/rep28b.status"

bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t28_full/2999"     "$RD/prior_t28.npz" 5 8007 0 "$RD/cl_pin_t28_full.json"     libero_object
bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t28_full/2999" NONE               5 8007 0 "$RD/cl_scratch_t28_full.json" libero_object
echo "REP28B_DONE" >> "$RD/rep28b.status"
