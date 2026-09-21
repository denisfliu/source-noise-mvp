#!/bin/bash
# Bad sketches on the real-only checkpoints (2026-09-20; Denis: "let's do the bad sketches test").
# The 4-click compounds whose own clearance to the gate cloud is 0.07 m (L->C) and 0.04 m (R->C):
#   pin0   gate_pin_joint_realonly, sketch at sigma 0        (sketch_cmpl_min4 / cmpr_min4)
#   pin05  same, sigma_serve 0.5                              (sketch_cmpl_min4s / cmpr_min4s)
#   vproj  gate_scratch_real, command in the source, v projected off U (exact carry, no slack)
#   sde05  gate_scratch_real, SDEdit t0 0.5
# 5 trials per cell; judge + clearance + tracking.
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy; CKROOT=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3
CK_SCR=$CKROOT/gate_scratch_real/4999; CK_PIN=$CKROOT/gate_pin_joint_realonly/4999
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_realonly.json"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
PORT=${PORT:-9160}; OUT=$RUN/badsketch_scores.txt; rm -f $RUN/badsketch.done $OUT
cd $RD
killport () { for p in $(ss -ltnp | grep ":$PORT " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
cell () { # tag mode sketch side scene prompt
  local TAG=$1 MODE=$2 SK=$3 SIDE=$4 SCENE=$5 PROMPT=$6
  killport
  case $MODE in
    pin)   setsid $EV $PINENV SNMVP_PIN_PROMPT=$RD/$SK CLOG=$RUN/clog_${TAG}.npy $GPU $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK_PIN --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    vproj) setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_VPROJ=1 $GPU $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --sketch $RD/$SK --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    sde:*) setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 $GPU $VENVPY $RD/serve_gate_sdedit.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --sketch $RD/$SK --t0 ${MODE#sde:} --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
  esac
  for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=14 APC=50 TRIALS=5 VIDEO=0 PROMPT="$PROMPT" \
    TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}.log 2>&1
  killport
  { echo "== badsketch: $TAG (mode=$MODE sketch=$SK scene=$SCENE)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side $SIDE
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
    $TV $RD/sketch_track.py --sketch $RD/$SK --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
cell bs_pin0_L  pin     sketch_cmpl_min4.json  left  left_and_center  "$CMP_L"
cell bs_pin05_L pin     sketch_cmpl_min4s.json left  left_and_center  "$CMP_L"
cell bs_vproj_L vproj   sketch_cmpl_min4.json  left  left_and_center  "$CMP_L"
cell bs_sde05_L sde:0.5 sketch_cmpl_min4.json  left  left_and_center  "$CMP_L"
cell bs_pin0_R  pin     sketch_cmpr_min4.json  right right_and_center "$CMP_R"
cell bs_pin05_R pin     sketch_cmpr_min4s.json right right_and_center "$CMP_R"
cell bs_vproj_R vproj   sketch_cmpr_min4.json  right right_and_center "$CMP_R"
cell bs_sde05_R sde:0.5 sketch_cmpr_min4.json  right right_and_center "$CMP_R"
echo DONE > $RUN/badsketch.done
