#!/bin/bash
# SDEdit (t0 0.5) and velocity projection on the plain real-only pi0, on exactly the sketches the real-only pin
# flies in run_realonly_reloc_hwsk.sh: the 12 relocated-gate poses (5 each) and the two hardware routes (10 each).
# 25-step replan, 0.30 of the card. Denis, 2026-09-24: "how do we compare to sdedit and the velocity projection".
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
U=$RD/pin_U_mh16.npy; CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch_real/4999; PORT=9021
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
OUT=$RUN/sketch_baselines_scores.txt; : > $OUT
kill_port () { for p in $(ss -ltnp | grep ":$1 " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
serve () {  # $1 mode $2 sketch $3 seed
  kill_port $PORT
  case $1 in
    sde)   setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_NOISE_SEED=$3 $GPU $VENVPY $RD/serve_gate_sdedit.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --sketch $2 --t0 0.5 --port $PORT >> $RUN/sv_sketchbase.log 2>&1 </dev/null & disown;;
    vproj) setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_VPROJ=1 SNMVP_NOISE_SEED=$3 $GPU $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --sketch $2 --port $PORT >> $RUN/sv_sketchbase.log 2>&1 </dev/null & disown;;
  esac
  for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT "
}
cd $RD
for MODE in sde vproj; do
  for SPEC in "fig8_denis3 left_and_center left 32" "orbit_wide right right 28"; do
    set -- $SPEC; SK=$1; SCENE=$2; SIDE=$3; NCH=$4; TAG=${MODE}25_$SK
    serve $MODE $RD/sketch_$SK.json 11 || { echo "SERVER_TIMEOUT $TAG" >> $OUT; continue; }
    env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=$NCH APC=25 TRIALS=10 VIDEO=0 TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_$TAG.log 2>&1
    kill_port $PORT
    { echo "== $TAG"; $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy; } >> $OUT 2>&1
    echo "CELL_DONE $TAG"
  done
  S=40
  for TF in -45,0,0 -25,0,0 25,0,0 45,0,0 90,0,0 0,0.5,-0.3 30,-0.4,0.4 100,-1.26,1.50 180,0,0 -35,1.34,0.75 90,0.74,1.95 90,-0.26,0.85; do
    S=$((S + 1)); SKT=rr25mg$(echo $TF | tr ',.-' '__m')_$S; TAG=${MODE}25mg$(echo $TF | tr ',.-' '__m')_$S
    IFS=, read DY DX DYY <<< "$TF"
    [ -f $RD/sketch_mg_$SKT.json ] || $VENVPY $RD/moved_gate_cell.py --make --dyaw $DY --dx $DX --dy $DYY --tag $SKT >> $OUT 2>&1
    serve $MODE $RD/sketch_mg_$SKT.json $S || { echo "SERVER_TIMEOUT $TAG" >> $OUT; continue; }
    env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right NCH=28 APC=25 TRIALS=5 VIDEO=0 GATE_TF=$TF TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_$TAG.log 2>&1
    kill_port $PORT
    { $VENVPY $RD/moved_gate_cell.py --score --dyaw $DY --dx $DX --dy $DYY --tag $TAG --traj $RUN/traj_${TAG}_*.npy; $TV $RD/gate_clearance.py --scene right --gate-tf=$TF --traj $RUN/traj_${TAG}_*.npy | tail -1; } >> $OUT 2>&1
    echo "CELL_DONE $TAG"
  done
done
echo CHAIN_DONE
