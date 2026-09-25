#!/bin/bash
# SDEdit (t0 0.5) and velocity projection on the plain real-only pi0, on fig8_denis3_cx15 (centre-gate crossing moved
# +0.15 m in x), 10 flights each, same seed and replan as ours (Denis, 2026-09-25).
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
U=$RD/pin_U_mh16.npy; CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch_real/4999; PORT=9021
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
SK=fig8_denis3_cx15
kill_port () { for p in $(ss -ltnp | grep ":$1 " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
for MODE in sde vproj; do
  TAG=${MODE}25_$SK; kill_port $PORT
  case $MODE in
    sde)   setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_NOISE_SEED=11 $GPU $VENVPY $RD/serve_gate_sdedit.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --sketch $RD/sketch_$SK.json --t0 0.5 --port $PORT >> $RUN/sv_$TAG.log 2>&1 </dev/null & disown;;
    vproj) setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_VPROJ=1 SNMVP_NOISE_SEED=11 $GPU $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --sketch $RD/sketch_$SK.json --port $PORT >> $RUN/sv_$TAG.log 2>&1 </dev/null & disown;;
  esac
  for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG"; continue; }
  cd $RD; env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=left SCENE=left_and_center NCH=32 APC=25 TRIALS=10 VIDEO=0 \
    TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_$TAG.log 2>&1
  kill_port $PORT
  $TV $RD/route_contact.py --scene left_and_center --sketch $RD/sketch_$SK.json --traj $RUN/traj_${TAG}_*.npy > $RUN/${TAG}_contact.txt 2>&1
  echo "CELL_DONE $TAG"
done
echo CHAIN_DONE
