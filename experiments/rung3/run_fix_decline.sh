#!/usr/bin/env bash
# Address the pin's late-step decline by fixing the prior mismatch: the step-sweep
# adapted on all 45 demos but used a prior fit on only 10. Refit the prior on the full
# 45 demos, then re-evaluate the EXISTING step checkpoints (no retraining) with the
# better prior. If the 0.98->0.66 decline flattens, the decline was prior error, and a
# consistent prior fixes it. GPU 0 only.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/fixdecline.status"

# 1) refit the prior on the full 45 demos (CPU)
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 SNMVP_EPISODES="$RD/fs_t21_full.json" \
  SNMVP_PRIOR_OUT="$RD/prior_t21_full.npz" SNMVP_NB=60 \
  $UV run python make_prior.py > "$RD/prior_t21_full.log" 2>&1
echo "PRIOR_REFIT $(grep -o 'R\^2 [0-9.]*' "$RD/prior_t21_full.log" | tail -1)" >> "$RD/fixdecline.status"

# 2) re-eval each step checkpoint with the refit prior (GPU 0, 30 trials)
for S in 500 1000 1500 2000 2500 2999; do
  bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21_steps/$S" "$RD/prior_t21_full.npz" 4 8001 0 \
    "$RD/cl_pin_t21_s${S}_refit.json" libero_object 30
  echo "REFIT_STEP_$S done" >> "$RD/fixdecline.status"
done
echo "FIXDECLINE_DONE" >> "$RD/fixdecline.status"
