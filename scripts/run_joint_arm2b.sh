#!/bin/bash
# run_joint_arm.sh with the pin basis as an argument (that script hardcodes pin_U_gate_rrr_k5 and is
# currently executing two arms, so it must not be edited — this is the successor, new file by rule).
#
#   run_joint_arm2.sh NAME GPU PORT STEPS DECAY TRIALS UPATH "EXTRA_ENV" [TRAIN_CFG]
#
# UPATH      pin basis .npy (e.g. pin_U_vla_base_k5.npy for the C arms)
# EXTRA      extra env for trainer AND readout AND server, e.g.
#            "SNMVP_HEAD_DETACH=0 SNMVP_FLOW_DETACH=1" for C2 (the two must travel together:
#            pi0.py refuses FLOW_DETACH without a coupled head, since nothing would train the VLM)
# TRAIN_CFG  training config (default pi0_gate; pi0_gate_freezevlm for C1). Readout/serve always
#            use pi0_gate — freezing only changes the optimizer mask, not the param structure.
set -u
NAME=$1; GPU=$2; PORT=$3; STEPS=$4; DECAY=$5; TRIALS=$6; UPATH=$7; EXTRA=${8:-}; TRAIN_CFG=${9:-pi0_gate}; SEED=${10:-42}
RUN=/home/ubuntu/ctxrun
RD=/home/ubuntu/code/source-noise-mvp/experiments/rung3
PY=/home/ubuntu/code/openpi/.venv/bin/python; HFB=/home/ubuntu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$UPATH
EXP=gate_pin_joint_$NAME
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
export PATH=/tmp/tv/bin:/usr/local/cuda-12.8/bin:$PATH; export CUDA_HOME=/usr/local/cuda-12.8
rm -f $RUN/arm_$NAME.done $RUN/arm_${NAME}_scores.txt
[ -f "$U" ] || { echo U_MISSING > $RUN/arm_$NAME.done; exit 1; }

for k in $(seq 1 480); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$u" -lt 20000 ] && break; sleep 30
done
cd /home/ubuntu/code/openpi

# 1. train. save-interval=STEPS so only the final checkpoint is written (the default 1000 filled the disk)
env -u VIRTUAL_ENV $BASE $EXTRA XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=$GPU \
  .venv/bin/python scripts/train.py $TRAIN_CFG --exp-name=$EXP --num-train-steps=$STEPS \
  --lr-schedule.decay-steps=$DECAY --save-interval=$STEPS --seed=$SEED --no-wandb-enabled --overwrite \
  > $RUN/arm_${NAME}_train.log 2>&1
FINAL=$((STEPS - 1))
CK=/home/ubuntu/code/openpi/checkpoints/$TRAIN_CFG/$EXP/$FINAL
[ -d "$CK/params" ] || { echo TRAIN_FAILED > $RUN/arm_$NAME.done; exit 1; }
rm -rf /home/ubuntu/code/openpi/checkpoints/$TRAIN_CFG/$EXP/*/train_state
{ echo "== arm $NAME  steps=$STEPS decay=$DECAY U=$(basename $U) cfg=$TRAIN_CFG extra='$EXTRA'"
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
# batch client: scene loads once per side, sides run in parallel (inference is ~10% of wall time,
# so two clients on one server stack cleanly). Claim tier keeps VIDEO=1 (stride-4 frames, fps 9).
for side in left right; do
  env CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT SIDE=$side SCENE=$side NCH=8 APC=50 TRIALS=$TRIALS \
    OUT=$RUN/overlay_arm${NAME}_${side}_{t}.mp4 TRAJ=$RUN/traj_arm${NAME}_${side}_{t}.npy \
    /tmp/tv/bin/python $RD/gate_rollout_batch.py > $RUN/roll_arm${NAME}_${side}.log 2>&1 &
  sleep 120  # stagger: the first client's handshake+compile blocks the server loop; a simultaneous second handshake times out (cost C2 its right side, 2026-08-13)
done
wait
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
{ for side in left right; do
    echo "== arm $NAME, APC=50, $side"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $PY $RD/gate_success.py \
      --traj $RUN/traj_arm${NAME}_${side}_*.npy --side $side
    /tmp/tv/bin/python $RD/gate_clearance.py --scene $side --traj $RUN/traj_arm${NAME}_${side}_*.npy
  done } >> $RUN/arm_${NAME}_scores.txt 2>&1
grep -qa "clearance-clean" $RUN/arm_${NAME}_scores.txt && echo DONE > $RUN/arm_$NAME.done \
  || echo SCORE_FAILED > $RUN/arm_$NAME.done
