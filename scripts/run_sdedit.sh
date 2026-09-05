#!/bin/bash
# SDEdit baseline for sketch commands (2026-09-02): the unpinned pi0 (scratch3, seed 42) with
# the IDENTICAL sketch pipeline, sketch used SDEdit-style (partial denoising from the whole
# sketch chunk at t0). Hand-drawn L->C compound (the ablation-1 control sketch; pin: 5/5+5/5,
# scratch+pin-injection: 0/5) at t0 in {0.3,0.5,0.7,0.9}; orbit at {0.3,0.5,0.7}. 5 trials/cell.
# Gated on the redrawn-sketch xswap batch (xsk3) finishing.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch3/4999
SRV=serve_gate_sdedit.py
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
RIGHT="go through the gate on the right and hover over the stuffed animal"
PORT=9140
OUT=$RUN/sdedit_scores.txt
rm -f $RUN/sdedit.done $OUT
for k in $(seq 1 240); do [ -f $RUN/xsk3.done ] && break; sleep 60; done
[ -f $RUN/xsk3.done ] || { echo GATE_TIMEOUT > $RUN/sdedit.fail; exit 1; }
cd $RD
cell () { # tag t0 sketch side scene prompt
  local TAG=$1 T0=$2 SK=$3 SIDE=$4 SCENE=$5 PROMPT=$6
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/$SRV --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --sketch $RD/$SK \
    --t0 $T0 --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    PROMPT="$PROMPT" TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_${TAG}.log 2>&1
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { echo "== sdedit baseline: $TAG (scratch3, sketch=$SK, t0=$T0)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
      --traj $RUN/traj_${TAG}_*.npy --side $SCENE
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
    $TV $RD/sketch_track.py --sketch $RD/$SK --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
for T in 0.3 0.5 0.7 0.9; do cell sde_cmpl_t${T/./} $T sketch_cmpl_denis.json left left_and_center "$CMP_L"; done
for T in 0.3 0.5 0.7;     do cell sde_orbit_t${T/./} $T sketch_orbit.json right right "$RIGHT"; done
echo DONE > $RUN/sdedit.done
