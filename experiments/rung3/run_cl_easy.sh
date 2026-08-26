#!/usr/bin/env bash
# Closed-loop eval on the easy held-out libero_object tasks: task 21 -> object task 4
# (GPU 0), task 28 -> object task 5 (GPU 1); each pin+prior then scratch.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
rm -f "$RD/cl_easy.status"

( bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21/799"     "$RD/prior_t21.npz" 4 8001 0 "$RD/cl_pin_t21.json"     libero_object
  bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t21/799" NONE               4 8001 0 "$RD/cl_scratch_t21.json" libero_object
  echo "G0_DONE" >> "$RD/cl_easy.status" ) &
( bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t28/799"     "$RD/prior_t28.npz" 5 8002 1 "$RD/cl_pin_t28.json"     libero_object
  bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t28/799" NONE               5 8002 1 "$RD/cl_scratch_t28.json" libero_object
  echo "G1_DONE" >> "$RD/cl_easy.status" ) &
wait
echo "CL_EASY_DONE" >> "$RD/cl_easy.status"
