#!/bin/bash
# Flawed sketches through the DEMONSTRATED left and right gates (2026-09-25; Denis: translate the course-correction
# argument to the left/right gates). sketch_bad_<side>[s].json: the usual route with the crossing shifted so the line
# passes ~0.065 m from a post (every other leg >= 0.33 m). 25-step replan, 10 flights per cell, seed 11. Tags bg25_*.
# The 4-click compounds whose own clearance to the gate cloud is 0.07 m (L->C) and 0.04 m (R->C):
#   pin0   gate_pin_joint_realonly, sketch at sigma 0        (sketch_cmpl_min4 / cmpr_min4)
#   pin05  same, sigma_serve 0.5                              (sketch_cmpl_min4s / cmpr_min4s)
#   vproj  gate_scratch_real, command in the source, v projected off U (exact carry, no slack)
#   sde05  gate_scratch_real, SDEdit t0 0.5
# Scored with score_badgate.py.
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy; CKROOT=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3
CK_SCR=$CKROOT/gate_scratch_real/4999; CK_PIN=$CKROOT/gate_pin_joint_realonly/4999
PINENV="SNMVP_NOISE_SEED=11 SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_realonly.json"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
PORT=${PORT:-9020}; OUT=$RUN/badgate25_scores.txt; rm -f $RUN/badgate25.done $OUT
cd $RD
killport () { for p in $(ss -ltnp | grep ":$PORT " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
cell () { # tag mode sketch side scene prompt
  local TAG=$1 MODE=$2 SK=$3 SIDE=$4 SCENE=$5 PROMPT=$6
  killport
  case $MODE in
    pin)   setsid $EV $PINENV SNMVP_PIN_PROMPT=$RD/$SK CLOG=$RUN/clog_${TAG}.npy $GPU $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK_PIN --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    vproj) setsid $EV SNMVP_NOISE_SEED=11 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_VPROJ=1 $GPU $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --sketch $RD/$SK --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    sde:*) setsid $EV SNMVP_NOISE_SEED=11 SNMVP_ZERO_PAD_ACTIONS=1 $GPU $VENVPY $RD/serve_gate_sdedit.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --sketch $RD/$SK --t0 ${MODE#sde:} --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
  esac
  for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=16 APC=25 TRIALS=10 VIDEO=0 PROMPT="$PROMPT" \
    TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}.log 2>&1
  killport
  { echo "== badsketch: $TAG (mode=$MODE sketch=$SK scene=$SCENE)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side $SIDE
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
    $TV $RD/sketch_track.py --sketch $RD/$SK --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
for side in left right; do
  P="go through the gate on the $side and hover over the stuffed animal"
  cell bg25_pin0_$side  pin     sketch_bad_$side.json  $side $side "$P"
  cell bg25_pin05_$side pin     sketch_bad_${side}s.json $side $side "$P"
  cell bg25_vproj_$side vproj   sketch_bad_$side.json  $side $side "$P"
  cell bg25_sde05_$side sde:0.5 sketch_bad_$side.json  $side $side "$P"
done
echo DONE > $RUN/badgate25.done
