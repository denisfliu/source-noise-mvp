#!/usr/bin/env bash
# Claim-based 2-GPU scheduler for the three gate training patterns. Two workers (GPU1 immediately,
# GPU0 once the head-to-head frees it) pull from a shared job list; each job is claimed atomically
# via mkdir, so they never collide and either worker resumes leftovers after a drop. norm stats + U
# are assumed done by run_gate_all.sh (skipped if present). Launch:
#   setsid bash run_gate_sched.sh </dev/null >gate_sched_driver.log 2>&1 & disown
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_gate
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
ST=$RD/gate_all.status
echo "SCHED_START $(date -u +%H:%M:%S)" >> "$ST"

ORDER="gate_synth_pin gate_synth_scratch gate_both_pin"
eps_of() { case "$1" in
  gate_synth_pin|gate_synth_scratch) echo "$RD/gate_synth_eps.json" ;;
  gate_both_pin) echo "$RD/gate_synth_realtrain.json" ;; esac; }
u_of() { case "$1" in gate_synth_scratch) echo "" ;; *) echo "$RD/pin_U_gate_k5.npy" ;; esac; }

worker() {  # gpu
  local gpu=$1 e ep u
  if [ "$gpu" = 0 ]; then
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 6000 ]; do sleep 60; done
    echo "GPU0 free, worker starting $(date -u +%H:%M:%S)" >> "$ST"
  fi
  for e in $ORDER; do
    [ -d "$CK/$e/4999" ] && continue
    mkdir "$RD/.claim_$e" 2>/dev/null || continue      # atomic claim; other worker skips
    [ -d "$CK/$e/4999" ] && continue
    ep=$(eps_of "$e"); u=$(u_of "$e")
    if [ -n "$u" ]; then export SNMVP_PIN_U="$u"; else unset SNMVP_PIN_U; fi
    echo "TRAIN_START $e gpu=$gpu $(date -u +%H:%M:%S)" >> "$ST"
    CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled SNMVP_EPISODES="$ep" \
      $UV run scripts/train.py pi0_gate --exp-name="$e" --num-train-steps=5000 --save-interval=2500 --overwrite > "$RD/$e.log" 2>&1
    echo "TRAIN_DONE $e gpu=$gpu rc=$? $(date -u +%H:%M:%S)" >> "$ST"
    unset SNMVP_PIN_U
  done
}

worker 1 &
worker 0 &
wait
echo "GATE_SCHED_ALL_DONE $(date -u +%H:%M:%S)" >> "$ST"
