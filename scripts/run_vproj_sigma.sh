#!/bin/bash
# Partial velocity projection sweep (2026-09-20): v <- (I-UU^T)v + s UU^T v on gate_scratch_real with the sketch in
# the source; s in {0.1, 0.3, 0.5}; orbit, fig8 (right), hand-drawn compound (left+center); 5 trials each.
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy; CK_SCR=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch_real/4999
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
RP="go through the gate on the right and hover over the stuffed animal"
PORT=${PORT:-9160}; OUT=$RUN/vproj_sigma_scores.txt; rm -f $RUN/vproj_sigma.done $OUT
cd $RD
killport () { for p in $(ss -ltnp | grep ":$PORT " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
cell () { # tag sigma sketch side scene prompt
  local TAG=$1 S=$2 SK=$3 SIDE=$4 SCENE=$5 PROMPT=$6
  killport
  setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_VPROJ=1 SNMVP_VPROJ_SIGMA=$S $GPU $VENVPY $RD/serve_gate_plain_sketch.py \
    --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --sketch $RD/$SK --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=14 APC=50 TRIALS=5 VIDEO=0 PROMPT="$PROMPT" \
    TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}.log 2>&1
  killport
  { echo "== vproj sigma $S: $TAG (sketch=$SK scene=$SCENE)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side $SIDE
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
    $TV $RD/sketch_track.py --sketch $RD/$SK --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
for S in 0.1 0.3 0.5; do
  T=vps${S/./}
  cell ${T}_orbit $S sketch_orbit.json right right "$RP"
  cell ${T}_fig8  $S sketch_fig8.json  right right "$RP"
  cell ${T}_cmpl  $S sketch_cmpl_denis.json left left_and_center "$CMP_L"
done
echo DONE > $RUN/vproj_sigma.done
