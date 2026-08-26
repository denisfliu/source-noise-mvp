#!/usr/bin/env bash
# Re-evaluate the EXISTING step-sweep checkpoints with 50 trials each (full
# libero_object init-state set) to tighten the success-vs-steps learning curve.
# No retraining. GPU 0 only, serial; robust to SSH drops (status-file driven).
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
rm -f "$RD/stepsweep_n50.status"

for S in 500 1000 1500 2000 2500 2999; do
  if [ -d "$CK/fs_pin_t21_steps/$S" ]; then
    bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21_steps/$S" "$RD/prior_t21.npz" 4 8001 0 \
      "$RD/cl_pin_t21_s${S}_n50.json" libero_object 50
  fi
  if [ -d "$CK/fs_scratch_t21_steps/$S" ]; then
    bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t21_steps/$S" NONE 4 8001 0 \
      "$RD/cl_scratch_t21_s${S}_n50.json" libero_object 50
  fi
  echo "STEP_${S}_DONE" >> "$RD/stepsweep_n50.status"
done
echo "STEPSWEEP_N50_DONE" >> "$RD/stepsweep_n50.status"
