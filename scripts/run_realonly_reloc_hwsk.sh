#!/bin/bash
# Real-only pin on (1) the 12 distinct arbitrary gate poses of 2026-08-28/29 (right gate moved in the splat, auto sketch
# through each pose; 5 trials each) and (2) the two sketches flown on hardware (fig8_denis3 through the left and
# centre gates, orbit_wide around the right gate; 10 trials each). 25-step replan, sigma 0 (from the sketch), 0.30
# of the card beside the port-8900 hardware server. Denis, 2026-09-24.
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
U=$RD/pin_U_mh16.npy; CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_realonly/4999; PORT=9020
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_realonly.json"
OUT=$RUN/realonly_reloc_hwsk_scores.txt; : > $OUT
kill_port () { for p in $(ss -ltnp | grep ":$1 " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
serve () {   # $1 sketch json, $2 seed
  kill_port $PORT
  setsid $EV $PINENV SNMVP_PIN_PROMPT=$1 SNMVP_NOISE_SEED=$2 XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT \
    >> $RUN/sv_realonly_reloc.log 2>&1 </dev/null & disown
  for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT "
}
cd $RD
# (2) hardware sketches first
for SPEC in; do
  set -- $SPEC; SK=$1; SCENE=$2; SIDE=$3; NCH=$4; TAG=ro25_$SK
  serve $RD/sketch_$SK.json 11 || { echo "SERVER_TIMEOUT $TAG" >> $OUT; continue; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=$NCH APC=25 TRIALS=10 VIDEO=0 \
    TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_$TAG.log 2>&1
  kill_port $PORT
  { echo "== $TAG"; $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy; $TV $RD/sketch_track.py --sketch $RD/sketch_$SK.json --traj $RUN/traj_${TAG}_*.npy; } >> $OUT 2>&1
  echo "CELL_DONE $TAG"
done
# (1) 12 relocated-gate poses (the 2026-08-29 list reran -45 and (0.5,-0.3) with the router; distinct poses = 12)
S=40
for TF in -45,0,0 -25,0,0 25,0,0 45,0,0 90,0,0 0,0.5,-0.3 30,-0.4,0.4 100,-1.26,1.50 180,0,0 -35,1.34,0.75 90,0.74,1.95 90,-0.26,0.85; do
  S=$((S + 1))
  TAG=rr25mg$(echo $TF | tr ',.-' '__m')_$S
  IFS=, read DY DX DYY <<< "$TF"
  $VENVPY $RD/moved_gate_cell.py --make --dyaw $DY --dx $DX --dy $DYY --tag $TAG >> $OUT 2>&1
  serve $RD/sketch_mg_$TAG.json $S || { echo "SERVER_TIMEOUT $TAG" >> $OUT; continue; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right NCH=28 APC=25 TRIALS=5 VIDEO=0 \
    GATE_TF=$TF TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_$TAG.log 2>&1
  kill_port $PORT
  $VENVPY $RD/moved_gate_cell.py --score --dyaw $DY --dx $DX --dy $DYY --tag $TAG --traj $RUN/traj_${TAG}_*.npy >> $OUT 2>&1
  echo "CELL_DONE $TAG"
done
echo CHAIN_DONE
