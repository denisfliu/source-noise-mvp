#!/bin/bash
# Local-4090 successor of run_joint_arm3.sh (box scripts kept for provenance; this machine has ONE
# GPU, different roots, and the renderer venv rebuilt at ~/code/tv). Full six-cell joint arm:
# train -> readout gate -> left/right (TRIALS, video) -> center CFL/CFR (10, VIDEO=0) ->
# compound screens (5, VIDEO=0). Sequential by necessity except the paired eval clients, which
# keep the 120 s stagger (simultaneous handshakes cost C2 its right side, 2026-08-13).
#
#   run_joint_arm_local.sh NAME PORT STEPS DECAY TRIALS UPATH "EXTRA_ENV" [TRAIN_CFG] [SEED]
set -u
NAME=$1; PORT=$2; STEPS=$3; DECAY=$4; TRIALS=$5; UPATH=$6; EXTRA=${7:-}; CFG=${8:-pi0_gate}; SEED=${9:-42}
GPU=0
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
OPI=/home/dfliu/code/openpi-snmvp
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=$OPI/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$UPATH
EXP=gate_pin_joint_$NAME
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
mkdir -p $RUN
rm -f $RUN/arm_$NAME.done $RUN/arm_${NAME}_scores.txt $RUN/ctr_${NAME}_scores.txt
[ -f "$U" ] || { echo U_MISSING > $RUN/arm_$NAME.done; exit 1; }

for k in $(seq 1 480); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$u" -lt 2000 ] && break; sleep 30
done

# 1. train (checkpoints land in $OPI/checkpoints/$CFG/$EXP; only the final step is saved)
cd $OPI
env -u VIRTUAL_ENV PYTHONPATH=$OPI/src $BASE $EXTRA XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  CUDA_VISIBLE_DEVICES=$GPU $VENVPY scripts/train.py $CFG --exp-name=$EXP \
  --num-train-steps=$STEPS --lr-schedule.decay-steps=$DECAY --save-interval=$STEPS \
  --seed=$SEED --no-wandb-enabled --overwrite > $RUN/arm_${NAME}_train.log 2>&1
FINAL=$((STEPS - 1))
CK=$OPI/checkpoints/$CFG/$EXP/$FINAL
[ -d "$CK/params" ] || { echo TRAIN_FAILED > $RUN/arm_$NAME.done; exit 1; }
rm -rf $OPI/checkpoints/$CFG/$EXP/*/train_state
{ echo "== arm $NAME  steps=$STEPS decay=$DECAY U=$(basename $U) cfg=$CFG seed=$SEED extra='$EXTRA'"
  grep -a "loss=" $RUN/arm_${NAME}_train.log | tail -2; } > $RUN/arm_${NAME}_scores.txt

# 2. readout gate: the head must reproduce the oracle c it was trained on, or serving is broken
cd $RD
HS=""; case "$EXTRA" in *SNMVP_HEAD_STATE=1*) HS="--head-state";; esac
$EV $BASE $EXTRA CUDA_VISIBLE_DEVICES=$GPU $VENVPY $RD/joint_head.py --ckpt $CK --pin-u $U \
  --norm $HFB/assets/gate_nav --n 8 $HS --check > $RUN/arm_${NAME}_check.log 2>&1
python3 - "$RUN/arm_${NAME}_check.log" >> $RUN/arm_${NAME}_scores.txt << 'PYEOF'
import sys
rows = [l.split() for l in open(sys.argv[1]) if l.startswith(("left", "right", "center"))]
r2 = [float(r[2]) for r in rows if len(r) >= 3]
print(f"== readout gate: {len(r2)} tasks, min per-task c-R2 {min(r2) if r2 else float('nan'):+.4f}")
print("== GATE PASS" if r2 and min(r2) > 0.5 else "== GATE FAIL")
PYEOF
grep -qa "GATE PASS" $RUN/arm_${NAME}_scores.txt || { echo GATE_FAILED > $RUN/arm_$NAME.done; exit 1; }

SRV=serve_gate_pin_joint.py
serve_up () {  # port clog-suffix
  for p in $(pgrep -f "$SRV --ckpt .* --port $1"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  # 0.45 of the card for the JAX server: the two torch render clients colocate on the same GPU
  setsid $EV $BASE $EXTRA CLOG=$RUN/clog_${NAME}$2.npy XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=$GPU $VENVPY $RD/$SRV \
    --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $1 \
    >> $RUN/sv_arm_$NAME.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$1 " && break; sleep 3; done
  ss -ltn | grep -q ":$1 "
}
serve_down () { for p in $(pgrep -f "$SRV --ckpt .* --port $1"); do kill -9 "$p" 2>/dev/null; done; }

# 3. left/right, claim-tier config (VIDEO=1), 120 s client stagger
serve_up $PORT "" || { echo SERVER_TIMEOUT > $RUN/arm_$NAME.done; exit 1; }
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
  done } >> $RUN/arm_${NAME}_scores.txt 2>&1
grep -qa "clearance-clean" $RUN/arm_${NAME}_scores.txt || { echo SCORE_FAILED > $RUN/arm_$NAME.done; exit 1; }

# 4. center + compounds (the addon cells), fresh server on PORT+1
P2=$((PORT + 1))
CFL="go through the center gate from the left and hover over the stuffed animal"
CFR="go through the center gate from the right and hover over the stuffed animal"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
serve_up $P2 "_ctr" || { echo CTR_SERVER_TIMEOUT > $RUN/arm_$NAME.done; exit 1; }
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
  done } > $RUN/ctr_${NAME}_scores.txt 2>&1
grep -qa "clearance-clean" $RUN/ctr_${NAME}_scores.txt && echo DONE > $RUN/arm_$NAME.done \
  || echo CTR_SCORE_FAILED > $RUN/arm_$NAME.done
