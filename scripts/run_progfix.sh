#!/bin/bash
# Does the head read the language, and can an out-of-distribution instruction be fixed after training?
# (2026-09-20; Denis: "the compound sentence is supposed to be OOD... is it possible to modify the GMM
# post training so it does better? language seems to be ignored.")
# ALL CELLS USE THE SAME SCENE (left_and_center) and the same start, so only the sentence differs.
#   A left        trained sentence, first leg
#   B cfl         trained sentence, second leg
#   C compound    the out-of-distribution sentence (control; head-only)
#   D program     the same OOD task run as a PROGRAM of the two trained sentences, advanced by the head's
#                 own declared intent (|net displacement of U c| < tau for two replans). No retraining.
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy; CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_gmsig3/4999
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_gmsig3.json SNMVP_CLOG_FULL=1"
L="go through the gate on the left and hover over the stuffed animal"
F="go through the center gate from the left and hover over the stuffed animal"
C="go through the gate on the left, then through the center gate and hover over the stuffed animal"
PORT=${PORT:-9160}; NCH=${NCH:-18}; TRIALS=${TRIALS:-5}
OUT=$RUN/progfix_scores.txt; rm -f $RUN/progfix.done $OUT
cd $RD
killport () { for p in $(ss -ltnp | grep ":$PORT " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
cell () { # tag prompt program_json
  local TAG=$1 PROMPT=$2 PROG=${3:-}
  killport
  [ -n "$PROG" ] && export SNMVP_PROMPT_PROGRAM="$PROG" || unset SNMVP_PROMPT_PROGRAM
  setsid $EV $PINENV CLOG=$RUN/clog_${TAG}.npy $GPU $VENVPY $RD/serve_gate_pin_joint.py \
    --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=left SCENE=left_and_center NCH=$NCH APC=50 TRIALS=$TRIALS VIDEO=0 PROMPT="$PROMPT" \
    TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}.log 2>&1
  killport
  { echo "== progfix: $TAG"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side left_and_center
    $TV $RD/gate_clearance.py --scene left_and_center --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
PROGJSON="$(python3 -c "import json,sys;print(json.dumps([sys.argv[1],sys.argv[2]]))" "$L" "$F")"
cell pf_prog "$C" "$PROGJSON"
export SNMVP_PROG_GATE=left
cell pf_proggate "$C" "$PROGJSON"
unset SNMVP_PROG_GATE
echo DONE > $RUN/progfix.done
