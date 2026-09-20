#!/bin/bash
# Three ways to put a sketch into a flow, on the REAL-ONLY checkpoints, in the simulator (2026-09-20; Denis:
# "inject the pin but also zero out the velocity component along the command; compare SDEdit, this, and ours").
#   sde:T0  SDEdit on gate_scratch_real: start at t0 from t0 z + (1-t0) a_sketch, unmodified flow
#   inject  gate_scratch_real with the sketch command written into the source noise, unmodified flow
#   vproj   same injection, and v <- (I - U U^T) v at every Euler step (SNMVP_VPROJ=1): the command is carried
#   ours    gate_pin_joint_realonly, sketch through the pin at sigma 0 (orbit/fig8 reuse the 2026-09-15 rollouts)
# Cells: orbit and fig8 (right scene), hand-drawn compound cmpl_denis (left+center). 5 trials each.
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy; CKROOT=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3
CK_SCR=$CKROOT/gate_scratch_real/4999; CK_PIN=$CKROOT/gate_pin_joint_realonly/4999
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_realonly.json"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
PORT=${PORT:-9160}; OUT=$RUN/vproj_scores_rf.txt; rm -f $RUN/vproj.done
cd $RD
killport () { for p in $(ss -ltnp | grep ":$PORT " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
cell () { # tag mode sketch side scene prompt
  local TAG=$1 MODE=$2 SK=$3 SIDE=$4 SCENE=$5 PROMPT=$6
  killport
  case $MODE in
    sde:*)  setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 $GPU $VENVPY $RD/serve_gate_sdedit.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --sketch $RD/$SK --t0 ${MODE#sde:} --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    inject) setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 $GPU $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --sketch $RD/$SK --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    vproj)  setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_VPROJ=1 $GPU $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --sketch $RD/$SK --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    ours)   setsid $EV $PINENV SNMVP_PIN_PROMPT=$RD/$SK CLOG=$RUN/clog_${TAG}.npy $GPU $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK_PIN --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
  esac
  for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=14 APC=50 TRIALS=5 VIDEO=0 PROMPT="$PROMPT" \
    TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}.log 2>&1
  killport
  { echo "== vproj: $TAG (mode=$MODE sketch=$SK scene=$SCENE)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side $SIDE
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
    $TV $RD/sketch_track.py --sketch $RD/$SK --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
RP="go through the gate on the right and hover over the stuffed animal"
for M in sde:0.5 inject vproj; do
  T=${M/:/}; T=${T/./}
  cell vp_${T}_orbit $M sketch_orbit.json right right "$RP"
  cell vp_${T}_fig8  $M sketch_fig8.json  right right "$RP"
done
for M in; do
  T=${M/:/}; T=${T/./}
  cell vp_${T}_cmpl $M sketch_cmpl_denis.json left left_and_center "$CMP_L"
done
echo DONE > $RUN/vproj.done
