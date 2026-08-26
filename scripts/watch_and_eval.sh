#!/usr/bin/env bash
# Watch a training run's checkpoint dir; evaluate each new checkpoint on
# LIBERO (GPU 1) as it appears. Stops after the final checkpoint
# (num_train_steps-1) has been evaluated.
#
# Usage: watch_and_eval.sh <run_ckpt_dir> <final_step> <trials_per_task> <results_dir> <run_tag>
set -uo pipefail

RUN_DIR="$1"; FINAL_STEP="$2"; TRIALS="$3"; RESULTS="$4"; TAG="$5"
EVAL="$(dirname "$0")/eval_checkpoint.sh"
mkdir -p "$RESULTS"

while true; do
  for d in "$RUN_DIR"/*/; do
    step=$(basename "$d")
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    out="$RESULTS/${TAG}_step${step}.json"
    [ -f "$out" ] && continue
    [ -f "$d/model.safetensors" ] || continue
    echo "WATCH: evaluating step $step ($(date -u +%H:%M:%S))"
    bash "$EVAL" "$d" "$TRIALS" "$out" || echo "WATCH: eval of step $step FAILED"
  done
  if [ -f "$RESULTS/${TAG}_step${FINAL_STEP}.json" ]; then
    echo "WATCH_FINAL=done (final step $FINAL_STEP evaluated)"
    exit 0
  fi
  sleep 300
done
