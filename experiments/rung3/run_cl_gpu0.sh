#!/usr/bin/env bash
# Serial closed-loop evals on GPU 0 ONLY (GPU 1 reserved for Denis's cosmos job).
# Waits for the last adaptation checkpoint, then evaluates all four combos one at a
# time on the easy held-out libero_object tasks (21 -> object 4, 28 -> object 5).
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
rm -f "$RD/cleval.status"

until [ -d "$CK/fs_scratch_t28/799" ]; do sleep 30; done
sleep 20   # let the adapt process release GPU 0

bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21/799"     "$RD/prior_t21.npz" 4 8001 0 "$RD/cl_pin_t21.json"     libero_object
bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t21/799" NONE               4 8001 0 "$RD/cl_scratch_t21.json" libero_object
bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t28/799"     "$RD/prior_t28.npz" 5 8002 0 "$RD/cl_pin_t28.json"     libero_object
bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t28/799" NONE               5 8002 0 "$RD/cl_scratch_t28.json" libero_object
echo "CLEVAL_DONE" >> "$RD/cleval.status"
