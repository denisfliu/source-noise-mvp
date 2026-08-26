#!/usr/bin/env bash
# Driver: 4 closed-loop evals. Task 0 (LIBERO_10 task 4) on GPU 0, task 1 (task 6) on
# GPU 1, each running pin then scratch. pin arm serves with its prior; scratch unpinned.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
rm -f "$RD/cl_all.status"

( bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t0/799"     "$RD/prior_t0.npz" 4 8001 0 "$RD/cl_pin_t0.json"
  bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t0/799" NONE              4 8001 0 "$RD/cl_scratch_t0.json"
  echo "GPU0_DONE" >> "$RD/cl_all.status" ) &

( bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t1/799"     "$RD/prior_t1.npz" 6 8002 1 "$RD/cl_pin_t1.json"
  bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t1/799" NONE              6 8002 1 "$RD/cl_scratch_t1.json"
  echo "GPU1_DONE" >> "$RD/cl_all.status" ) &

wait
echo "CL_ALL_DONE" >> "$RD/cl_all.status"
