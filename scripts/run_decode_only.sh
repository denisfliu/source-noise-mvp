#!/bin/bash
# Decode-only diagnostic (2026-09-04, Denis: "what do trajectories look like if we just inject the pin
# and nothing else"): execute U c per replan with no denoising. c from the xswap head (CFR, left) or
# from the hand-drawn L->C sketch (compound). 5 trials each.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_xswap.json SNMVP_PIN_DECODE_ONLY=1"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999
PORT=9220
OUT=$RUN/decode_only_scores.txt
rm -f $RUN/decode_only.done $OUT
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
cd $RD
cell () { # tag extra_env side scene judge prompt
  local TAG=$1 EXTRA=$2 SIDE=$3 SCENE=$4 JS=$5 PROMPT=$6
  for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done; sleep 3
  setsid $EV $PINENV $EXTRA CLOG=$RUN/clog_${TAG}.npy $GPU $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate \
    --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    PROMPT="$PROMPT" TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}.log 2>&1
  for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { echo "== decode-only: $TAG"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side $JS
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
cell dec_cfr  ""  right center center_from_right "go through the center gate from the right and hover over the stuffed animal"
cell dec_left ""  left  left   left              "go through the gate on the left and hover over the stuffed animal"
cell dec_cmpl "SNMVP_PIN_PROMPT=$RD/sketch_cmpl_denis.json" left left_and_center left_and_center "$CMP_L"
echo DONE > $RUN/decode_only.done
