#!/bin/bash
# TRIGGER-FREE nudges for the compound (2026-09-03, Denis: no trigger of any kind): the advice is
# target points consumed by proximity from the first replan; the task prompt is never touched unless
# swap_prompt. Pin (xswap s42) x {2 targets coarse / coarse+swap / h50 / all16, 1 target coarse,
# 1 target idle-gated}; SDEdit t0=0.3 (scratch3) with the 2-target pursuit chunk as guide.
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
PORT=9170
OUT=$RUN/nudge_scores.txt
rm -f $RUN/nudge.done $OUT
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
  { echo "== nudge: $TAG (mode=$MODE advice=$AJ)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side left_and_center
    $TV $RD/gate_clearance.py --scene left_and_center --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
cell ng_2t_coarse      pin     nudge_cmpl_2t_coarse.json
cell ng_2t_h50         pin     nudge_cmpl_2t_h50.json
cell ng_2t_all         pin     nudge_cmpl_2t_all.json
cell ng_2t_coarse_swap pin     nudge_cmpl_2t_coarse_swap.json
cell ng_1t_coarse      pin     nudge_cmpl_1t_coarse.json
cell ng_1t_idle        pin     nudge_cmpl_1t_idle.json
cell ng_sde03_2t       sde:0.3 nudge_cmpl_2t_all.json
echo DONE > $RUN/nudge.done
