#!/bin/bash
# Minimal command-space ADVICE for the compound task (2026-09-03, Denis): after the gate-1 transit,
# a target point (the center-gate exit) becomes a pursuit command and only the named command
# coordinates override the head's. Pin (xswap s42): none (prompt swap only) / all / coarse_xy /
# h50_xy / two targets coarse_xy. SDEdit t0=0.3 (scratch3) with the same pursuit chunk as guide.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_xswap.json"
CK_PIN=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999
CK_SCR=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch3/4999
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
PORT=9160
OUT=$RUN/advice_scores.txt
rm -f $RUN/advice.done $OUT
cd $RD
killport () { for p in $(pgrep -f "port $PORT"); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
cell () { # tag mode(pin|sde:T0) advice_json
  local TAG=$1 MODE=$2 AJ=$3
  killport
  if [ "$MODE" = pin ]; then
    setsid $EV $PINENV SNMVP_PIN_ADVICE=$RD/$AJ CLOG=$RUN/clog_${TAG}.npy $GPU \
      $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK_PIN --config pi0_gate --norm $HFB/assets/gate_nav \
      --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  else
    setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 $GPU \
      $VENVPY $RD/serve_gate_sdedit.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav \
      --sketch $RD/$AJ --advice --t0 ${MODE#sde:} --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  fi
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=left SCENE=left_and_center NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    PROMPT="$CMP_L" TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}.log 2>&1
  killport
  { echo "== advice: $TAG (mode=$MODE advice=$AJ)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side left_and_center
    $TV $RD/gate_clearance.py --scene left_and_center --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
cell adv_none   pin     advice_cmpl_none.json
cell adv_coarse pin     advice_cmpl_coarse.json
cell adv_h50    pin     advice_cmpl_h50.json
cell adv_all    pin     advice_cmpl_all.json
cell adv_2t     pin     advice_cmpl_2t.json
cell adv_sde03  sde:0.3 advice_cmpl_all.json
echo DONE > $RUN/advice.done
