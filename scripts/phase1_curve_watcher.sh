#!/usr/bin/env bash
# Per-arm success/adherence-vs-steps curves: watch all phase1_* runs and
# evaluate every new 2500-step checkpoint.
#   arms A, B -> 10-trial success eval (canonical states)  [mid-schedule caveat]
#   arm C     -> offline wrong-invariant probe (success needs the oracle; the
#                plain protocol is meaningless for always-pinned models)
# Idempotent: skips checkpoints whose output JSON already exists.
set -uo pipefail

REPO="$HOME/code/source-noise-mvp"
RESULTS="$REPO/experiments/phase1/results"
CKROOT="$HOME/code/openpi/checkpoints/pi0_libero"
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
export HF_HOME="$HOME/code/hf-cache" HF_LEROBOT_HOME="$HOME/code/hf-cache/lerobot"

while true; do
  for run_dir in "$CKROOT"/phase1_*; do
    [ -d "$run_dir" ] || continue
    exp=$(basename "$run_dir")
    arm=${exp#phase1_}; arm=${arm%%_*}
    for d in "$run_dir"/*/; do
      step=$(basename "$d")
      [[ "$step" =~ ^[0-9]+$ ]] || continue
      [ -f "$d/model.safetensors" ] || continue
      if [ "$arm" = "C" ]; then
        out="$RESULTS/${exp}_step${step}_probe.json"
        [ -f "$out" ] && continue
        echo "CURVE: probe $exp step $step ($(date -u +%H:%M:%S))"
        ( cd "$HOME/code/openpi" && CUDA_VISIBLE_DEVICES=1 uv run python \
            "$REPO/scripts/wrong_invariant_probe.py" \
            --checkpoint "$d" --num-samples 8 --noise-draws 3 --seed 0 \
            --out "$out" ) || echo "CURVE: probe $exp/$step FAILED"
      else
        out="$RESULTS/evals/${exp}_step${step}.json"
        [ -f "$out" ] && continue
        echo "CURVE: eval $exp step $step ($(date -u +%H:%M:%S))"
        SNMVP_EVAL_PORT=8030 bash "$REPO/scripts/eval_checkpoint.sh" "$d" 10 "$out" \
          || echo "CURVE: eval $exp/$step FAILED"
      fi
    done
  done
  sleep 600
done
