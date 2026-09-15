#!/bin/bash
# Fly any sketch on any pin arm in the simulator and score it (2026-09-15).
#   bash scripts/run_sketch_cell.sh <sketch name> <arm> [trials=5] [scene=right] [side=right]
#   sketch name -> experiments/rung3/sketch_<name>.json ; arm -> ours (xswapc) | ours_xswap | noswap (gmsig3) | realonly
# Outputs: ctxrun/traj_sk_<arm>_<name>_{t}.npy, ctxrun/sk_<arm>_<name>_scores.txt (judge + clearance + tracking),
# and viz/sk_<arm>_<name>.html (cloud page, built with build_traj_page.py --no-judge).
set -u
NAME=${1:?sketch name}; ARM=${2:?arm}; TRIALS=${3:-5}; SCENE=${4:-right}; SIDE=${5:-right}
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
CKROOT=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3; U=$RD/pin_U_mh16.npy; PORT=${PORT:-9123}
case $ARM in
  ours)       CK=$CKROOT/gate_pin_joint_xswapc/4999;   SIG=$RD/sigma_map_xswapc.json;;
  ours_xswap) CK=$CKROOT/gate_pin_joint_xswap/4999;    SIG=$RD/sigma_map_xswap.json;;
  noswap)     CK=$CKROOT/gate_pin_joint_gmsig3/4999;   SIG=$RD/sigma_map_gmsig3.json;;
  realonly)   CK=$CKROOT/gate_pin_joint_realonly/4999; SIG=$RD/sigma_map_realonly.json;;
  *) echo "arm must be ours | ours_xswap | noswap | realonly"; exit 2;;
esac
SK=$RD/sketch_$NAME.json; [ -f "$SK" ] || { echo "missing $SK"; exit 1; }
TAG=sk_${ARM}_$NAME; OUT=$RUN/${TAG}_scores.txt; rm -f $OUT
cd $RD
for p in $(pgrep -f "serve_gate_pin_joint.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done; sleep 2
setsid env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U \
  SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 \
  SNMVP_SIGMA_MAP=$SIG SNMVP_PIN_PROMPT=$SK XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_$TAG.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT >> $OUT; exit 1; }
env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=14 APC=50 TRIALS=$TRIALS VIDEO=0 \
  TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_$TAG.log 2>&1
for p in $(pgrep -f "serve_gate_pin_joint.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done
{ echo "== sketch $NAME on $ARM ($TRIALS trials, scene $SCENE)"
  env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side $SIDE
  $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
  echo "== sketch tracking"; /home/dfliu/miniforge3/bin/python3 $RD/sketch_track.py --sketch $SK --traj $RUN/traj_${TAG}_*.npy
} >> $OUT 2>&1
/home/dfliu/miniforge3/bin/python3 $RD/viz/build_traj_page.py --no-judge --out $TAG.html --title "Sketch $NAME on $ARM" \
  --section "$NAME on $ARM|$SCENE|$SK|$ARM=$TAG" --note "Sketch $NAME flown by the $ARM arm, $TRIALS trials, scene $SCENE." >> $OUT 2>&1
tail -25 $OUT
