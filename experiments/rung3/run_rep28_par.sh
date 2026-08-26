#!/usr/bin/env bash
# Parallelize the rest of task-28 replication. Pin adapt already runs on GPU 0; launch
# scratch adapt on GPU 1 now, wait for both checkpoints, then eval both in parallel.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
SCR=$CK/snmvp_src_scratch/4999/params
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/rep28par.status"

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  SNMVP_INIT_CKPT="$SCR" SNMVP_PIN_U="" SNMVP_EPISODES="$RD/fs_t28_full.json" \
  $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=fs_scratch_t28_full \
  --num-train-steps=3000 --save-interval=3000 --overwrite > "$RD/fs_scratch_t28_full.log" 2>&1 &

until [ -d "$CK/fs_pin_t28_full/2999" ] && [ -d "$CK/fs_scratch_t28_full/2999" ]; do sleep 30; done
sleep 20
echo "ADAPTS_DONE" >> "$RD/rep28par.status"

( bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t28_full/2999"     "$RD/prior_t28.npz" 5 8007 0 "$RD/cl_pin_t28_full.json"     libero_object ) &
( bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t28_full/2999" NONE               5 8008 1 "$RD/cl_scratch_t28_full.json" libero_object ) &
wait
echo "REP28PAR_DONE" >> "$RD/rep28par.status"
