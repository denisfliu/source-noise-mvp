#!/bin/bash
# Figure-8 sketch through the UNPINNED scratch3 flow (sketch pipeline identical to run_scratch_sketch.sh's
# orbit cell; 2026-09-07, Denis: "see if injecting that sketch into the scratch policy also produces the
# same maneuvers"). 5 trials, right scene, no prompt override (sketch's prompt_after applies).
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch3/4999
PORT=9130; TAG=scrsk_fig8
cd $RD
for p in $(pgrep -f "serve_gate_plain_sketch.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done; sleep 2
setsid env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src \
  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CK --norm $HFB/assets/gate_nav \
  --pin-u $RD/pin_U_mh16.npy --sketch $RD/sketch_fig8.json --port $PORT >> $RUN/sv_$TAG.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT >> $RUN/scrsk_fig8_scores.txt; echo DONE > $RUN/scrsk_fig8.done; exit 1; }
env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right NCH=14 APC=50 TRIALS=5 VIDEO=0 \
  TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_$TAG.log 2>&1
for p in $(pgrep -f "serve_gate_plain_sketch.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done
{ echo "== scratch-sketch $TAG"
  env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
    $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side right
  $TV $RD/gate_clearance.py --scene right --traj $RUN/traj_${TAG}_*.npy
} >> $RUN/scrsk_fig8_scores.txt 2>&1
echo DONE > $RUN/scrsk_fig8.done
