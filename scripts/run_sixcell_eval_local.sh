#!/bin/bash
# Eval-only six cells for an ALREADY-TRAINED joint checkpoint (local 4090). Stages 3-4 of
# run_joint_arm_local.sh without train/readout — for re-flying after an eval-side failure or
# re-measuring an old checkpoint. Readout gate is assumed already passed (run joint_head --check
# yourself if in doubt). Same conventions: APC=50, L/R at TRIALS with video, center 10 VIDEO=0,
# compounds 5 VIDEO=0, 120 s client stagger, server at 45% of the card.
#
#   run_sixcell_eval_local.sh NAME CKPT PORT TRIALS "EXTRA_ENV" [UPATH]
#
# Markers/outputs: ev6_$NAME.done, ev6_${NAME}_scores.txt (L/R), ev6_${NAME}_ctr_scores.txt.
set -u
NAME=$1; CK=$2; PORT=$3; TRIALS=$4; EXTRA=${5:-}; UPATH=${6:-}
GPU=0
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=${UPATH:-$RD/pin_U_gate_rrr_k5.npy}   # MUST match the checkpoint basis (K mismatch = restore crash, bit gmsig3 2026-08-24)
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
SRV=serve_gate_pin_joint.py
mkdir -p $RUN
rm -f $RUN/ev6_$NAME.done $RUN/ev6_${NAME}_scores.txt $RUN/ev6_${NAME}_ctr_scores.txt
[ -d "$CK/params" ] || { echo CKPT_MISSING > $RUN/ev6_$NAME.done; exit 1; }
[ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -ge 5 ] \
  || { echo DISK_FULL > $RUN/ev6_$NAME.done; exit 1; }
# stale-glob guard: scoring globs traj_arm${NAME}_* / traj_${NAME}_* below
rm -f $RUN/traj_arm${NAME}_*.npy $RUN/overlay_arm${NAME}_*.mp4 $RUN/traj_${NAME}_*.npy \
      $RUN/clog_${NAME}.npy $RUN/clog_${NAME}_ctr.npy

for k in $(seq 1 720); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$u" -lt 2000 ] && break; sleep 60
done
u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
[ "$u" -lt 2000 ] || { echo GPU_BUSY_TIMEOUT > $RUN/ev6_$NAME.done; exit 1; }
cd $RD

serve_up () {  # port clog-suffix
  for p in $(pgrep -f "$SRV --ckpt .* --port $1"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV $BASE $EXTRA CLOG=$RUN/clog_${NAME}$2.npy XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=$GPU $VENVPY $RD/$SRV \
    --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $1 \
    >> $RUN/sv_ev6_$NAME.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$1 " && break; sleep 3; done
  ss -ltn | grep -q ":$1 "
}
serve_down () { for p in $(pgrep -f "$SRV --ckpt .* --port $1"); do kill -9 "$p" 2>/dev/null; done; }

serve_up $PORT "" || { echo SERVER_TIMEOUT > $RUN/ev6_$NAME.done; exit 1; }
for side in left right; do
  env CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT SIDE=$side SCENE=$side NCH=8 APC=50 TRIALS=$TRIALS \
    OUT=$RUN/overlay_arm${NAME}_${side}_{t}.mp4 TRAJ=$RUN/traj_arm${NAME}_${side}_{t}.npy \
    $TV $RD/gate_rollout_batch.py > $RUN/roll_arm${NAME}_${side}.log 2>&1 &
  sleep 120
done
wait
serve_down $PORT
{ for side in left right; do
    echo "== arm $NAME, APC=50, $side"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
      --traj $RUN/traj_arm${NAME}_${side}_*.npy --side $side
    $TV $RD/gate_clearance.py --scene $side --traj $RUN/traj_arm${NAME}_${side}_*.npy
  done } >> $RUN/ev6_${NAME}_scores.txt 2>&1
grep -qa "clearance-clean" $RUN/ev6_${NAME}_scores.txt || { echo SCORE_FAILED > $RUN/ev6_$NAME.done; exit 1; }

P2=$((PORT + 1))
CFL="go through the center gate from the left and hover over the stuffed animal"
CFR="go through the center gate from the right and hover over the stuffed animal"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
serve_up $P2 "_ctr" || { echo CTR_SERVER_TIMEOUT > $RUN/ev6_$NAME.done; exit 1; }
roll () {  # tag side scene prompt nch trials
  env CUDA_VISIBLE_DEVICES=$GPU PORT=$P2 SIDE=$2 SCENE=$3 NCH=$5 APC=50 TRIALS=$6 VIDEO=0 \
    PROMPT="$4" TRAJ=$RUN/traj_${NAME}_$1_{t}.npy \
    $TV $RD/gate_rollout_batch.py > $RUN/roll_${NAME}_$1.log 2>&1
}
roll cfl left  center "$CFL" 10 10 & sleep 120
roll cfr right center "$CFR" 10 10 &
wait
# compounds run SEQUENTIALLY: the duplicated-gate splats are bigger (7.0 + 4.8 GB clients) and
# two of them do not fit beside the 11 GB server on 24 GB -- cmpr OOM'd in ctl's eval (2026-08-20)
roll cmpl left  left_and_center  "$CMP_L" 14 5
roll cmpr right right_and_center "$CMP_R" 14 5
serve_down $P2
{ for spec in "cfl center_from_left center" "cfr center_from_right center" \
              "cmpl left_and_center -" "cmpr right_and_center -"; do
    set -- $spec
    echo "== $NAME $1 (judge: $2)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
      --traj $RUN/traj_${NAME}_$1_*.npy --side $2
    if [ "$3" != "-" ]; then SCN=$3; else SCN=$2; fi
    $TV $RD/gate_clearance.py --scene $SCN --traj $RUN/traj_${NAME}_$1_*.npy
  done } > $RUN/ev6_${NAME}_ctr_scores.txt 2>&1
grep -qa "clearance-clean" $RUN/ev6_${NAME}_ctr_scores.txt && echo DONE > $RUN/ev6_$NAME.done \
  || echo CTR_SCORE_FAILED > $RUN/ev6_$NAME.done
