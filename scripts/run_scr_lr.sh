#!/bin/bash
# Scratch control on the plain left/right gates, same protocol as every other grid cell
# (APC=50, NCH=8, 10 trials/side, VIDEO=0 trajectory-only).
set -u
RUN=/home/ubuntu/ctxrun; GPU=0; PORT=8916
RD=/home/ubuntu/code/source-noise-mvp/experiments/rung3
PY=/home/ubuntu/code/openpi/.venv/bin/python
EV="env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false"
export PATH=/tmp/tv/bin:/usr/local/cuda-12.8/bin:$PATH; export CUDA_HOME=/usr/local/cuda-12.8
rm -f $RUN/scr_lr.done $RUN/scr_lr_scores.txt
cd $RD
setsid $EV CUDA_VISIBLE_DEVICES=$GPU $PY $RD/serve_gate_pin_classic.py --mode scratch --port $PORT \
  >> $RUN/sv_scr_lr.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT > $RUN/scr_lr.done; exit 1; }
for side in left right; do
  env CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT SIDE=$side SCENE=$side NCH=8 APC=50 TRIALS=10 VIDEO=0 \
    TRAJ=$RUN/traj_scr_${side}_{t}.npy /tmp/tv/bin/python $RD/gate_rollout_batch.py \
    > $RUN/roll_scr_$side.log 2>&1 &
done
wait
for p in $(pgrep -f "serve_gate_pin_classic.py --mode scratch --port $PORT"); do kill -9 "$p" 2>/dev/null; done
{ for side in left right; do
    echo "== scratch, APC=50, $side"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $PY $RD/gate_success.py \
      --traj $RUN/traj_scr_${side}_*.npy --side $side
    /tmp/tv/bin/python $RD/gate_clearance.py --scene $side --traj $RUN/traj_scr_${side}_*.npy
  done } > $RUN/scr_lr_scores.txt 2>&1
grep -qa "clearance-clean" $RUN/scr_lr_scores.txt && echo DONE > $RUN/scr_lr.done \
  || echo SCORE_FAILED > $RUN/scr_lr.done
