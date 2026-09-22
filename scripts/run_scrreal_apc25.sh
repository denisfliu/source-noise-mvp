#!/bin/bash
# Real-only plain pi0 (gate_scratch_real) at a 25-step replan (2026-09-22; Denis: "also check real only scratch with
# this same thing"). Mirrors run_realonly_apc25.sh with the plain server; same 400 steps, 10 trials, same scorer.
#   bash scripts/run_scrreal_apc25.sh [APC=25]
set -u
APC=${1:-25}; NCH=$((400 / APC)); NAME=scrreal; TAG=${NAME}_apc$APC
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch_real/4999
PORT=${PORT:-9021}
kill_port () { for p in $(ss -ltnp | grep ":$1 " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; }
cd $RD
kill_port $PORT; sleep 3
setsid $EV XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY $RD/serve_gate_plain.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --port $PORT \
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
