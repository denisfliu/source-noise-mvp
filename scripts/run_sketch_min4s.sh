#!/bin/bash
# Denis's hand-drawn FULL-ROUTE sketches (Sketchpad, 2026-08-25): one server per sketch
# (SNMVP_PIN_PROMPT is a server-level binding), cmpl then cmpr, 5 trials each, route-clean
# judge + clearance. The sketches start at the start box, so the sketch owns the whole
# flight at sigma=0 and the head takes over only for the goal hover at handback.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_mh16.npy
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
HEADENV="SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_gmsig3.json"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_gmsig3/4999
SRV=serve_gate_pin_joint.py
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
PORT=9062
rm -f $RUN/sketch_min4s.done $RUN/traj_skm4s_*.npy $RUN/sketch_min4s_scores.txt
cd $RD
run_cell() { # tag scene client_prompt
  local TAG=$1 SCENE=$2 CPROMPT=$3
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV $BASE $HEADENV SNMVP_PIN_PROMPT=$RD/sketch_${TAG}_min4s.json \
    CLOG=$RUN/clog_skm4s_${TAG}.npy \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/$SRV --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U \
    --port $PORT >> $RUN/sv_skm4s_${TAG}.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $RUN/sketch_min4s.done; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=${SCENE%%_*} SCENE=$SCENE NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    PROMPT="$CPROMPT" TRAJ=$RUN/traj_skm4s_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_skm4s_${TAG}.log 2>&1
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { echo "== gmsig3 + MINIMAL 4-point sketch sigma=0.5, $TAG (route-clean judge)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
      --traj $RUN/traj_skm4s_${TAG}_*.npy --side $SCENE
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_skm4s_${TAG}_*.npy
  } >> $RUN/sketch_min4s_scores.txt 2>&1
}
run_cell cmpl left_and_center "$CMP_L"
run_cell cmpr right_and_center "$CMP_R"
echo DONE >> $RUN/sketch_min4s.done
