#!/bin/bash
# Scratch-sketch mechanism control (2026-08-30): scratch3 through the identical sketch
# pipeline. Cells: hand-drawn CMPL (compound scene) + orbit (right). Gated on the
# ablation trainings finishing.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch3/4999
PORT=9130
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
rm -f $RUN/scrsketch.done $RUN/scrsketch_scores.txt
for k in $(seq 1 420); do [ -f $RUN/ablations.done ] && break; sleep 60; done
[ -f $RUN/ablations.done ] || { echo GATE_TIMEOUT > $RUN/scrsketch.fail; exit 1; }
cd $RD
cell () { # tag sketch side scene prompt
  local TAG=$1 SK=$2 SIDE=$3 SCENE=$4 PR=$5
  for p in $(pgrep -f "serve_gate_plain_sketch.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CK --norm $HFB/assets/gate_nav \
    --pin-u $RD/pin_U_mh16.npy --sketch $RD/$SK --port $PORT >> $RUN/sv_$TAG.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    ${PR:+PROMPT="$PR"} TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_$TAG.log 2>&1
  for p in $(pgrep -f "serve_gate_plain_sketch.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { echo "== scratch-sketch $TAG"
    env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu \
      CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side $SCENE
  } >> $RUN/scrsketch_scores.txt 2>&1
}
cell scrsk_cmpl sketch_cmpl_denis.json left left_and_center "$CMP_L"
cell scrsk_orbit sketch_orbit.json right right ""
echo DONE > $RUN/scrsketch.done
