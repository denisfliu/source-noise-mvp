#!/bin/bash
# Worse sketches (2026-09-20; Denis: "let's make the sketch worse"): the 4-click L->C with its center waypoint
# slid west so the line passes the center gate's west post at 0.02 m (w2), 0.009 m (w3), and on the WRONG side
# of the post (w4, 0.033 m outside the aperture). Mixed-data checkpoints: gmsig3 pin sigma 0 / 0.5; scratch3
# with projection s = 0 / 0.1 / 0.3; SDEdit t0 0.5. 5 trials each.
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy; CKROOT=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3
CK_SCR=$CKROOT/gate_scratch3/4999; CK_PIN=$CKROOT/gate_pin_joint_gmsig3/4999
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_gmsig3.json"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
PORT=${PORT:-9160}; OUT=$RUN/worsesketch_scores.txt; rm -f $RUN/worsesketch.done $OUT
cd $RD
killport () { for p in $(ss -ltnp | grep ":$PORT " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
cell () { # tag mode sketch
  local TAG=$1 MODE=$2 SK=$3 SIDE=left SCENE=left_and_center PROMPT="$CMP_L"
  killport
  case $MODE in
    pin)     setsid $EV $PINENV SNMVP_PIN_PROMPT=$RD/$SK CLOG=$RUN/clog_${TAG}.npy $GPU $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK_PIN --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    vproj:*) setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_VPROJ=1 SNMVP_VPROJ_SIGMA=${MODE#vproj:} $GPU $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --sketch $RD/$SK --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    sde:*)   setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 $GPU $VENVPY $RD/serve_gate_sdedit.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --sketch $RD/$SK --t0 ${MODE#sde:} --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
  esac
  for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=14 APC=50 TRIALS=5 VIDEO=0 PROMPT="$PROMPT" \
    TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}.log 2>&1
  killport
  { echo "== worsesketch: $TAG (mode=$MODE sketch=$SK)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side $SCENE
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
    $TV $RD/sketch_track.py --sketch $RD/$SK --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
for W in w2 w3 w4; do
  cell ws_${W}_pin0   pin        sketch_cmpl_min4_${W}.json
  cell ws_${W}_pin05  pin        sketch_cmpl_min4_${W}s.json
  cell ws_${W}_vp0    vproj:0    sketch_cmpl_min4_${W}.json
  cell ws_${W}_vp01   vproj:0.1  sketch_cmpl_min4_${W}.json
  cell ws_${W}_vp03   vproj:0.3  sketch_cmpl_min4_${W}.json
  cell ws_${W}_sde05  sde:0.5    sketch_cmpl_min4_${W}.json
done
echo DONE > $RUN/worsesketch.done
