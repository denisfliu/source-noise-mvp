#!/bin/bash
# Center + compound eval add-on for any JOINT checkpoint: the missing 4 of 6 cells (Denis,
# 2026-08-12: "why do you never do center scenes when you eval the models?"). Chains behind an
# arm's .done marker so C2/C1 (mid-flight under run_joint_arm2.sh, which only flies left/right)
# still get full six-cell coverage. Future arms use run_joint_arm3.sh which includes all six.
#
#   run_center_addon.sh NAME CKPT GPU PORT UPATH "EXTRA_ENV" [WAIT_DONE]
set -u
NAME=$1; CK=$2; GPU=$3; PORT=$4; UPATH=$5; EXTRA=${6:-}; WAIT=${7:-}
RUN=/home/ubuntu/ctxrun
RD=/home/ubuntu/code/source-noise-mvp/experiments/rung3
PY=/home/ubuntu/code/openpi/.venv/bin/python; HFB=/home/ubuntu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false"
SRV=serve_gate_pin_joint.py
export PATH=/tmp/tv/bin:/usr/local/cuda-12.8/bin:$PATH; export CUDA_HOME=/usr/local/cuda-12.8
CFL="go through the center gate from the left and hover over the stuffed animal"
CFR="go through the center gate from the right and hover over the stuffed animal"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
rm -f $RUN/ctr_$NAME.done $RUN/ctr_${NAME}_scores.txt
if [ -n "$WAIT" ]; then
  for k in $(seq 1 720); do [ -f "$WAIT" ] && break; sleep 60; done
  [ -f "$WAIT" ] || { echo WAIT_TIMEOUT > $RUN/ctr_$NAME.done; exit 1; }
fi
[ -d "$CK/params" ] || { echo CKPT_MISSING > $RUN/ctr_$NAME.done; exit 1; }
for k in $(seq 1 400); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$u" -lt 20000 ] && break; sleep 30
done
cd $RD
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
sleep 3
setsid $EV SNMVP_HEAD=1 SNMVP_PIN_U=$UPATH SNMVP_HEAD_DETACH=0 $EXTRA \
  CUDA_VISIBLE_DEVICES=$GPU $PY $RD/$SRV --ckpt $CK --config pi0_gate \
  --norm $HFB/assets/gate_nav --pin-u $UPATH --port $PORT \
  >> $RUN/sv_ctr_$NAME.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT > $RUN/ctr_$NAME.done; exit 1; }
roll () {  # tag side scene prompt nch trials
  env CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT SIDE=$2 SCENE=$3 NCH=$5 APC=50 TRIALS=$6 VIDEO=0 \
    PROMPT="$4" TRAJ=$RUN/traj_${NAME}_$1_{t}.npy \
    /tmp/tv/bin/python $RD/gate_rollout_batch.py > $RUN/roll_${NAME}_$1.log 2>&1
}
roll cfl left  center "$CFL" 10 10 &
roll cfr right center "$CFR" 10 10 &
wait
roll cmpl left  left_and_center  "$CMP_L" 14 5 &
roll cmpr right right_and_center "$CMP_R" 14 5 &
wait
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
{ for spec in "cfl center_from_left center" "cfr center_from_right center" \
              "cmpl left_and_center -" "cmpr right_and_center -"; do
    set -- $spec
    echo "== $NAME $1 (judge: $2)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $PY $RD/gate_success.py \
      --traj $RUN/traj_${NAME}_$1_*.npy --side $2
    if [ "$3" != "-" ]; then SCN=$3; else SCN=$2; fi
    /tmp/tv/bin/python $RD/gate_clearance.py --scene $SCN --traj $RUN/traj_${NAME}_$1_*.npy
  done } > $RUN/ctr_${NAME}_scores.txt 2>&1
grep -qa "clearance-clean" $RUN/ctr_${NAME}_scores.txt && echo DONE > $RUN/ctr_$NAME.done \
  || echo SCORE_FAILED > $RUN/ctr_$NAME.done
