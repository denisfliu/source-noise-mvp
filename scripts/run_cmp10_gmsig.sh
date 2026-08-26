#!/bin/bash
# Compound promotion re-fly (2026-08-22): CMPL + CMPR at 10 trials WITH VIDEO on both
# sigma-conditioned checkpoints (gmsig seed 42, gmsigs7 seed 7). Purpose: gmsigs7's CMPL 5/5
# both-gates+dwell was a 5-trial VIDEO=0 screen — the first learned-arm compound completions
# ever if they survive 10 trials + clearance + human video. Each arm serves with ITS OWN
# calibration map (sigma* is checkpoint-specific). Sequential everything (compound splats are
# the big clients). Scores: cmp10_<arm>_scores.txt; marker cmp10.done.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_mh16.npy
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
HEADENV="SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1"
SRV=serve_gate_pin_joint.py
PORT=8998
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
rm -f $RUN/cmp10.done
cd $RD
run_arm () {  # name ckpt
  local NAME=$1 CK=$2
  local EXTRA="$HEADENV SNMVP_SIGMA_MAP=$RD/sigma_map_$NAME.json"
  rm -f $RUN/cmp10_${NAME}_scores.txt $RUN/traj_c10${NAME}_*.npy $RUN/overlay_c10${NAME}_*.mp4
  for k in $(seq 1 240); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    [ "$u" -lt 2000 ] && break; sleep 30
  done
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV $BASE $EXTRA CLOG=$RUN/clog_c10$NAME.npy XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 $VENVPY $RD/$SRV \
    --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT \
    >> $RUN/sv_c10$NAME.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $NAME" >> $RUN/cmp10.done; return 1; }
  for spec in "cmpl left left_and_center" "cmpr right right_and_center"; do
    set -- $spec
    local PR; if [ "$1" = "cmpl" ]; then PR="$CMP_L"; else PR="$CMP_R"; fi
    env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$2 SCENE=$3 NCH=14 APC=50 TRIALS=10 VIDEO=1 \
      OUT=$RUN/overlay_c10${NAME}_$1_{t}.mp4 PROMPT="$PR" TRAJ=$RUN/traj_c10${NAME}_$1_{t}.npy \
      $TV $RD/gate_rollout_batch.py > $RUN/roll_c10${NAME}_$1.log 2>&1
  done
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { for spec in "cmpl left_and_center" "cmpr right_and_center"; do
      set -- $spec
      echo "== $NAME $1 x10 VIDEO (judge: $2)"
      $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
        --traj $RUN/traj_c10${NAME}_$1_*.npy --side $2
      $TV $RD/gate_clearance.py --scene $2 --traj $RUN/traj_c10${NAME}_$1_*.npy
    done } >> $RUN/cmp10_${NAME}_scores.txt 2>&1
}
run_arm gmsigs7 /home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate/gate_pin_joint_gmsigs7/4999
run_arm gmsig   /home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate/gate_pin_joint_gmsig/4999
echo DONE >> $RUN/cmp10.done
