#!/bin/bash
# Live-intent cockpit session (2026-08-29): xswap server with the intent bridge + a
# 3-trial right-cell client. Open experiments/rung3/viz/live_intent_right.html in a
# browser FIRST (it reconnects automatically), then run this.
#   run_live_intent.sh [TRIALS=3] [SKETCH=]   (SKETCH=sketch json name for sketch flights)
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
U=$RD/pin_U_mh16.npy
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999
TRIALS=${1:-3}
SKETCH=${2:-}
PORT=9110
for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
sleep 2
cd $RD
setsid env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 \
  SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 \
  SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 \
  SNMVP_SIGMA_MAP=$RD/sigma_map_xswap.json SNMVP_INTENT_WS=8765 \
  ${SKETCH:+SNMVP_PIN_PROMPT=$RD/$SKETCH} \
  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav \
  --pin-u $U --port $PORT >> $RUN/sv_liveintent.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT; exit 1; }
echo "server up — cockpit should show 'live'. Flying $TRIALS trials..."
env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right NCH=14 APC=50 TRIALS=$TRIALS VIDEO=0 \
  TRAJ=$RUN/traj_live_{t}.npy $TV $RD/gate_rollout_batch.py 2>&1 | grep -a "trial\|THROUGH"
for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
echo done
