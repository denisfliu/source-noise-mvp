#!/bin/bash
# Relocated-gate poses given as "dyaw,dx,dy seed" arguments (2026-09-25, Denis: random far-away poses, out of the
# cluster at the original right gate: >= 1.2 m from takeoff, >= 1.0 m from the goal, >= 0.8 m between gate centres,
# sketch >= 0.25 m from the gate). Ours, SDEdit and velocity projection, 5 flights each.
#   bash run_reloc_replace.sh "-163,0.04,-1.09 51" "174,2.81,1.37 52" ...
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
U=$RD/pin_U_mh16.npy; CKP=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_realonly/4999
CKS=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch_real/4999; PORT=9020
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_realonly.json"
kill_port () { for p in $(ss -ltnp | grep ":$1 " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
cd $RD
for PS in "$@"; do
  set -- $PS; TF=$1; S=$2; PT=$(echo $TF | tr ',.-' '__m')_$S; IFS=, read DY DX DYY <<< "$TF"
  $VENVPY $RD/moved_gate_cell.py --make --dyaw=$DY --dx=$DX --dy=$DYY --tag rr25mg$PT
  SK=$RD/sketch_mg_rr25mg$PT.json
  for ARM in rr25mg sde25mg vproj25mg; do
    TAG=$ARM$PT; kill_port $PORT
    case $ARM in
      rr25mg)    setsid $EV $PINENV SNMVP_PIN_PROMPT=$SK SNMVP_NOISE_SEED=$S $GPU $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CKP --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_reloc_replace.log 2>&1 </dev/null & disown;;
      sde25mg)   setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_NOISE_SEED=$S $GPU $VENVPY $RD/serve_gate_sdedit.py --ckpt $CKS --config pi0_gate --norm $HFB/assets/gate_nav --sketch $SK --t0 0.5 --port $PORT >> $RUN/sv_reloc_replace.log 2>&1 </dev/null & disown;;
      vproj25mg) setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_VPROJ=1 SNMVP_NOISE_SEED=$S $GPU $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CKS --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --sketch $SK --port $PORT >> $RUN/sv_reloc_replace.log 2>&1 </dev/null & disown;;
    esac
    for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
    ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG"; continue; }
    env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right NCH=28 APC=25 TRIALS=5 VIDEO=0 GATE_TF=$TF \
      TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_$TAG.log 2>&1
    kill_port $PORT
    echo "CELL_DONE $TAG"
  done
done
echo CHAIN_DONE
