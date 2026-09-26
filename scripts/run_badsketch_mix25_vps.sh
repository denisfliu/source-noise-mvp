#!/bin/bash
# Partial velocity projection on the center-gate proof of concept (2026-09-25; Denis: try different sigmas for velocity
# projection): v <- (I - UU^T) v + s UU^T v on gate_scratch3, s in {0.1, 0.3, 0.5}. Tags bsm25_vps<s>_L. Based on
# run_badsketch_mix25.sh:
# the 4-click L->C sketch (0.07 m from the center post), gmsig3 at sigma 0 / 0.5, velocity projection and SDEdit on
# gate_scratch3, 25-step replan, 10 flights each, seed 11. Tags bsm25_*. Scored with score_badsketch.py (PREFIX=bsm25).
# The 4-click compounds whose own clearance to the gate cloud is 0.07 m (L->C) and 0.04 m (R->C):
# 10 trials per cell; scored with score_badsketch.py (route_contact + gate_success.judge_compound gates, no dwell).
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy; CKROOT=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3
CK_SCR=$CKROOT/gate_scratch3/4999; CK_PIN=$CKROOT/gate_pin_joint_gmsig3/4999
PINENV="SNMVP_NOISE_SEED=11 SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_gmsig3.json"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
PORT=${PORT:-9020}; OUT=$RUN/badsketchmix25vps_scores.txt; rm -f $RUN/badsketchmix25vps.done $OUT
cd $RD
killport () { for p in $(ss -ltnp | grep ":$PORT " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
cell () { # tag mode sketch side scene prompt
  local TAG=$1 MODE=$2 SK=$3 SIDE=$4 SCENE=$5 PROMPT=$6
  killport
  case $MODE in
    pin)   setsid $EV $PINENV SNMVP_PIN_PROMPT=$RD/$SK CLOG=$RUN/clog_${TAG}.npy $GPU $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK_PIN --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    vps:*) setsid $EV SNMVP_NOISE_SEED=11 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_VPROJ=1 SNMVP_VPROJ_SIGMA=${MODE#vps:} $GPU $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --sketch $RD/$SK --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    vproj) setsid $EV SNMVP_NOISE_SEED=11 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_VPROJ=1 $GPU $VENVPY $RD/serve_gate_plain_sketch.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --sketch $RD/$SK --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
    sde:*) setsid $EV SNMVP_NOISE_SEED=11 SNMVP_ZERO_PAD_ACTIONS=1 $GPU $VENVPY $RD/serve_gate_sdedit.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav --sketch $RD/$SK --t0 ${MODE#sde:} --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown;;
  esac
  for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=28 APC=25 TRIALS=10 VIDEO=0 PROMPT="$PROMPT" \
    TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}.log 2>&1
  killport
  { echo "== badsketch: $TAG (mode=$MODE sketch=$SK scene=$SCENE)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side $SIDE
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
    $TV $RD/sketch_track.py --sketch $RD/$SK --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
for S in 0.1 0.3 0.5; do
  cell bsm25_vps${S/./}_L vps:$S sketch_cmpl_min4.json left left_and_center "$CMP_L"
done
echo DONE > $RUN/badsketchmix25vps.done
