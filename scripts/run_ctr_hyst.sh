#!/bin/bash
# Compound cells (CFL, CFR 10 trials; CMPL, CMPR 5 trials) for a joint-head checkpoint with the served-component
# pi-hysteresis margin set by HYST (2026-09-22, latch ablation). Same server env, prompts, judges and scene clouds
# as the ctr section of run_realonly_post.sh; runs on the 0.30 card slice beside the port-8900 hardware server.
#   HYST=0 bash scripts/run_ctr_hyst.sh <NAME>     -> ~/ctxrun/ctr_<NAME>_hyst<HYST>_scores.txt, traj_<NAME>_hyst<HYST>_<cell>_<t>.npy
set -u
NAME=${1:?name}; HYST=${HYST:-0.2}; TAG=${NAME}_hyst$HYST
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
U=$RD/pin_U_mh16.npy
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_$NAME.json SNMVP_GMM_HYST=$HYST"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_$NAME/4999
PORT=${PORT:-9021}
kill_port () { for p in $(ss -ltnp | grep ":$1 " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; }
cd $RD
kill_port $PORT; sleep 3
setsid $EV $PINENV CLOG=$RUN/clog_${TAG}_ctr.npy XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT \
  >> $RUN/sv_${TAG}_ctr.log 2>&1 </dev/null & disown
for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT; exit 1; }
CFL="go through the center gate from the left and hover over the stuffed animal"
CFR="go through the center gate from the right and hover over the stuffed animal"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
roll () {
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$2 SCENE=$3 NCH=$5 APC=50 TRIALS=$6 VIDEO=0 \
    PROMPT="$4" TRAJ=$RUN/traj_${TAG}_$1_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}_$1.log 2>&1
}
roll cfl left  center "$CFL" 10 10 & sleep 120
roll cfr right center "$CFR" 10 10 &
wait
roll cmpl left  left_and_center  "$CMP_L" 14 5
roll cmpr right right_and_center "$CMP_R" 14 5
kill_port $PORT
{ for spec in "cfl center_from_left center" "cfr center_from_right center" "cmpl left_and_center -" "cmpr right_and_center -"; do
    set -- $spec
    echo "== $TAG $1 (judge: $2)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_$1_*.npy --side $2
    if [ "$3" != "-" ]; then SCN=$3; else SCN=$2; fi
    $TV $RD/gate_clearance.py --scene $SCN --traj $RUN/traj_${TAG}_$1_*.npy
  done } > $RUN/ctr_${TAG}_scores.txt 2>&1
echo BATCH_ALL_DONE >> $RUN/ctr_${TAG}_scores.txt
