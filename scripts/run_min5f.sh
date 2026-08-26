#!/bin/bash
# Corrected min5 (true-aperture gate waypoint) on BOTH pin checkpoints (2026-08-26).
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
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
PORT=9084
rm -f $RUN/min5f.done $RUN/min5f_scores.txt
cd $RD
cell () { # tag ckpt sigmap
  local TAG=$1 CK=$2 SM=$3
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV $BASE $HEADENV SNMVP_SIGMA_MAP=$RD/$SM SNMVP_PIN_PROMPT=$RD/sketch_cmpr_min5f.json \
    CLOG=$RUN/clog_${TAG}.npy XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/$SRV --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U \
    --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $RUN/min5f_scores.txt; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right_and_center NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    PROMPT="$CMP_R" TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_${TAG}.log 2>&1
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { echo "== min5f (true-aperture waypoint): $TAG"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
      --traj $RUN/traj_${TAG}_*.npy --side right_and_center
    $TV $RD/gate_clearance.py --scene right_and_center --traj $RUN/traj_${TAG}_*.npy
  } >> $RUN/min5f_scores.txt 2>&1
}
cell m5f42_cmpr /home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_gmsig3/4999 sigma_map_gmsig3.json
cell m5fs7_cmpr /home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_gmsig3s7/4999 sigma_map_gmsig3s7.json
echo DONE > $RUN/min5f.done
