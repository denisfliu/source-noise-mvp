#!/bin/bash
# One parameterised joint-training arm: train -> readout gate -> fly -> score.
# Written as a single script taking arguments because deriving these by sed line-ranges has been the
# main source of self-inflicted breakage today (clobbered variable blocks, gates that silently did not
# match, a doubled exp name).
#
#   run_joint_arm.sh NAME GPU PORT STEPS DECAY TRIALS "EXTRA_ENV"
#
# NAME    experiment name suffix, e.g. b2 or b1long
# STEPS   training steps
# DECAY   lr_schedule.decay_steps. The stock config uses 1,000,000, so a 5k run holds the LR at
#         essentially peak the whole time and never anneals — matching DECAY to STEPS is the
#         "trained properly" condition.
# EXTRA   extra env for the trainer, e.g. "SNMVP_HEAD_DETACH=0" for the coupled variant
set -u
NAME=$1; GPU=$2; PORT=$3; STEPS=$4; DECAY=$5; TRIALS=$6; EXTRA=${7:-}
RUN=/home/ubuntu/ctxrun
RD=/home/ubuntu/code/source-noise-mvp/experiments/rung3
PY=/home/ubuntu/code/openpi/.venv/bin/python; HFB=/home/ubuntu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_gate_rrr_k5.npy
EXP=gate_pin_joint_$NAME
CK=/home/ubuntu/code/openpi/checkpoints/pi0_gate/$EXP/4999
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
export PATH=/tmp/tv/bin:/usr/local/cuda-12.8/bin:$PATH; export CUDA_HOME=/usr/local/cuda-12.8
rm -f $RUN/arm_$NAME.done $RUN/arm_${NAME}_scores.txt

for k in $(seq 1 480); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$u" -lt 20000 ] && break; sleep 30
done
cd /home/ubuntu/code/openpi

# 1. train. save-interval=STEPS so only the final checkpoint is written (the default 1000 filled the disk)
env -u VIRTUAL_ENV $BASE $EXTRA XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=$GPU \
  .venv/bin/python scripts/train.py pi0_gate --exp-name=$EXP --num-train-steps=$STEPS \
  --lr-schedule.decay-steps=$DECAY --save-interval=$STEPS --no-wandb-enabled --overwrite \
  > $RUN/arm_${NAME}_train.log 2>&1
FINAL=$((STEPS - 1))
CK=/home/ubuntu/code/openpi/checkpoints/pi0_gate/$EXP/$FINAL
[ -d "$CK/params" ] || { echo TRAIN_FAILED > $RUN/arm_$NAME.done; exit 1; }
rm -rf /home/ubuntu/code/openpi/checkpoints/pi0_gate/$EXP/*/train_state
{ echo "== arm $NAME  steps=$STEPS decay=$DECAY extra='$EXTRA'"
  grep -a "loss=" $RUN/arm_${NAME}_train.log | tail -2; } > $RUN/arm_${NAME}_scores.txt

# 2. readout gate: the head must reproduce the oracle c it was trained on, or serving is broken
cd $RD
HS=""; case "$EXTRA" in *SNMVP_HEAD_STATE=1*) HS="--head-state";; esac
$EV $BASE $EXTRA CUDA_VISIBLE_DEVICES=$GPU $PY $RD/joint_head.py --ckpt $CK --pin-u $U \
  --norm $HFB/assets/gate_nav --n 8 $HS --check > $RUN/arm_${NAME}_check.log 2>&1
python3 - "$RUN/arm_${NAME}_check.log" >> $RUN/arm_${NAME}_scores.txt << 'PYEOF'
import sys
rows = [l.split() for l in open(sys.argv[1]) if l.startswith(("left", "right", "center"))]
r2 = [float(r[2]) for r in rows if len(r) >= 3]
print(f"== readout gate: {len(r2)} tasks, min per-task c-R2 {min(r2) if r2 else float('nan'):+.4f}")
print("== GATE PASS" if r2 and min(r2) > 0.5 else "== GATE FAIL")
PYEOF
grep -qa "GATE PASS" $RUN/arm_${NAME}_scores.txt || { echo GATE_FAILED > $RUN/arm_$NAME.done; exit 1; }

# 3. fly, same client config as every other APC=50 arm
SRV=serve_gate_pin_joint.py
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
sleep 3
setsid $EV $BASE $EXTRA CLOG=$RUN/clog_$NAME.npy CUDA_VISIBLE_DEVICES=$GPU $PY $RD/$SRV \
  --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT \
  >> $RUN/sv_arm_$NAME.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT > $RUN/arm_$NAME.done; exit 1; }
for t in $(seq 1 "$TRIALS"); do
  for side in left right; do
    env CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT SIDE=$side SCENE=$side NCH=8 APC=50 \
      OUT=$RUN/overlay_arm${NAME}_${side}_$t.mp4 TRAJ=$RUN/traj_arm${NAME}_${side}_$t.npy \
      /tmp/tv/bin/python $RD/gate_video_overlay.py > $RUN/roll_arm${NAME}_${side}_$t.log 2>&1
  done
done
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
{ for side in left right; do
    echo "== arm $NAME, APC=50, $side"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $PY $RD/gate_success.py \
      --traj $RUN/traj_arm${NAME}_${side}_*.npy --side $side
    /tmp/tv/bin/python $RD/gate_clearance.py --scene $side --traj $RUN/traj_arm${NAME}_${side}_*.npy
  done } >> $RUN/arm_${NAME}_scores.txt 2>&1
grep -qa "clearance-clean" $RUN/arm_${NAME}_scores.txt && echo DONE > $RUN/arm_$NAME.done \
  || echo SCORE_FAILED > $RUN/arm_$NAME.done
