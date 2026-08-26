#!/bin/bash
# Post-training pipeline for gmsig (sigma-conditioned GMM x mh16, 2026-08-21): waits for the
# checkpoint, then readout gate -> sigma calibration probe on the NEW head (--save rows) ->
# build SNMVP_SIGMA_MAP -> six-cell eval served with the per-replan trained trust dial.
# Calibration must use this checkpoint's own head: sigma* distributions are checkpoint-specific.
set -u
NAME=gmsig3s7
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_mh16.npy
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
TRAINEXTRA="SNMVP_DATA_DIR=data_gate_synth3 SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_$NAME/4999
PORT=9012
SRV=serve_gate_pin_joint.py
rm -f $RUN/arm_$NAME.done
# RESUME (2026-08-26): gate+probe already done; map built manually after the
# stale data_gate_synth default crashed make_sigma_map (now fixed with --data-dir)
EXTRA="$TRAINEXTRA SNMVP_SIGMA_MAP=$RD/sigma_map_$NAME.json"

serve_up () {
  for p in $(pgrep -f "$SRV --ckpt .* --port $1"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV $BASE $EXTRA CLOG=$RUN/clog_${NAME}$2.npy XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 $VENVPY $RD/$SRV \
    --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $1 \
    >> $RUN/sv_arm_$NAME.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$1 " && break; sleep 3; done
  ss -ltn | grep -q ":$1 "
}
serve_down () { for p in $(pgrep -f "$SRV --ckpt .* --port $1"); do kill -9 "$p" 2>/dev/null; done; }

serve_up $PORT "" || { echo SERVER_TIMEOUT > $RUN/arm_$NAME.done; exit 1; }
for side in left right; do
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$side SCENE=$side NCH=8 APC=50 TRIALS=10 \
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

P2=$((PORT + 1))
CFL="go through the center gate from the left and hover over the stuffed animal"
CFR="go through the center gate from the right and hover over the stuffed animal"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
serve_up $P2 "_ctr" || { echo CTR_SERVER_TIMEOUT > $RUN/arm_$NAME.done; exit 1; }
roll () {
  env CUDA_VISIBLE_DEVICES=0 PORT=$P2 SIDE=$2 SCENE=$3 NCH=$5 APC=50 TRIALS=$6 VIDEO=0 \
    PROMPT="$4" TRAJ=$RUN/traj_${NAME}_$1_{t}.npy \
    $TV $RD/gate_rollout_batch.py > $RUN/roll_${NAME}_$1.log 2>&1
}
roll cfl left  center "$CFL" 10 10 & sleep 120
roll cfr right center "$CFR" 10 10 &
wait
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
