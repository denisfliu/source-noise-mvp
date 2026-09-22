#!/bin/bash
# Real-only pin at a 25-step replan (2026-09-22; Denis: "failures of real pin are kind of weird. can we try
# running 25 action chunks instead?"). Same checkpoint, server, sigma map and scorer as the APC-50 cells in
# run_realonly_post.sh; NCH doubled to 16 so the flight covers the same 400 steps. Left and right, 10 trials.
#   bash scripts/run_realonly_apc25.sh [NAME=realonly] [APC=25]
set -u
NAME=${1:-realonly}; APC=${2:-25}; NCH=$((400 / APC))
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
U=$RD/pin_U_mh16.npy
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_$NAME.json"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_$NAME/4999
PORT=${PORT:-9020}; TAG=${NAME}_apc$APC
kill_port () { for p in $(ss -ltnp | grep ":$1 " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; }
cd $RD
kill_port $PORT; sleep 3
# 0.30 of the card: the hardware server (port 8900, ~8.6 GB) stays up beside this
setsid $EV $PINENV CLOG=$RUN/clog_$TAG.npy XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT \
  >> $RUN/sv_$TAG.log 2>&1 </dev/null & disown
for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT; exit 1; }
for side in left right; do
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$side SCENE=$side NCH=$NCH APC=$APC TRIALS=10 VIDEO=0 \
    TRAJ=$RUN/traj_arm${TAG}_${side}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_arm${TAG}_${side}.log 2>&1 &
  sleep 120
done
wait
kill_port $PORT
{ for side in left right; do
    echo "== arm $NAME, APC=$APC NCH=$NCH, $side"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_arm${TAG}_${side}_*.npy --side $side
    $TV $RD/gate_clearance.py --scene $side --traj $RUN/traj_arm${TAG}_${side}_*.npy
  done } > $RUN/arm_${TAG}_scores.txt 2>&1
echo BATCH_ALL_DONE >> $RUN/arm_${TAG}_scores.txt
