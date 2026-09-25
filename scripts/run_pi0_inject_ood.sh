#!/bin/bash
# Plain pi0 (gate_scratch_real) with the sketch command injected into its source noise -- no command training, no
# velocity projection -- on the OOD sketches of Table 1 (2026-09-25, Denis: fill the pi0 OOD cells). Same sketches,
# seeds and 25-step replan as ours. Tags: inj25mg<pose>_<seed>, inj25_orbit_wide, inj25_fig8_denis3_cx15.
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
U=$RD/pin_U_mh16.npy; CKS=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch_real/4999; PORT=9020
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
kill_port () { for p in $(ss -ltnp | grep ":$1 " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
serve () {   # $1 sketch, $2 seed
  kill_port $PORT
  setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_NOISE_SEED=$2 $GPU $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CKS --config pi0_gate \
    --norm $HFB/assets/gate_nav --pin-u $U --sketch $1 --port $PORT >> $RUN/sv_inj25.log 2>&1 </dev/null & disown
  for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT "
}
cd $RD
for PS in "-163,0.04,-1.09 51" "174,2.81,1.37 52" "2,1.06,2.68 53" "153,2.46,-0.3 54" "-58,1.55,1.76 55" "132,1.69,-0.83 56" \
          "-49,-0.06,2.97 57" "44,2.03,2.6 58" "130,0.42,-0.36 59" "-86,2.1,0.87 60" "10,-1.73,2.16 61" "34,-2.6,2.2 62" "107,-1.29,0.24 63"; do
  set -- $PS; TF=$1; S=$2; PT=$(echo $TF | tr ',.-' '__m')_$S; TAG=inj25mg$PT
  serve $RD/sketch_mg_rr25mg$PT.json $S || { echo "SERVER_TIMEOUT $TAG"; continue; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right NCH=28 APC=25 TRIALS=5 VIDEO=0 GATE_TF=$TF \
    TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_$TAG.log 2>&1
  echo "CELL_DONE $TAG"
done
for SPEC in "orbit_wide right right 28" "fig8_denis3_cx15 left_and_center left 32"; do
  set -- $SPEC; TAG=inj25_$1
  serve $RD/sketch_$1.json 11 || { echo "SERVER_TIMEOUT $TAG"; continue; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$3 SCENE=$2 NCH=$4 APC=25 TRIALS=10 VIDEO=0 \
    TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_$TAG.log 2>&1
  echo "CELL_DONE $TAG"
done
kill_port $PORT
echo CHAIN_DONE
