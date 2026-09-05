#!/bin/bash
# Ablation B (2026-09-03): SDEdit on the UNPINNED scratch3 guided by OUR head's decoded command U c.
# The head runs as its own service on the xswap checkpoint. Relaunch of run_ablate_cmd.sh part B with
# smaller JAX memory fractions (0.32 head service + 0.38 flow ~ 16.5 GB) so the gsplat renderer keeps
# ~6 GB — at 0.42 + 0.42 the renderer OOM'd. CFR at t0 0.3 / 0.5, then left / right / CFL at 0.3; 10 trials.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_mh16.npy
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_xswap.json"
CK_PIN=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999
CK_SCR=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch3/4999
PORT=9210; HPORT=9200
OUT=$RUN/ablate_cmd_B_scores.txt
rm -f $RUN/ablate_cmd_B.done $OUT
declare -A PROMPT SIDE SCENE JS
PROMPT[left]="go through the gate on the left and hover over the stuffed animal";   SIDE[left]=left;  SCENE[left]=left;   JS[left]=left
PROMPT[right]="go through the gate on the right and hover over the stuffed animal"; SIDE[right]=right; SCENE[right]=right; JS[right]=right
PROMPT[cfl]="go through the center gate from the left and hover over the stuffed animal";  SIDE[cfl]=left;  SCENE[cfl]=center; JS[cfl]=center_from_left
PROMPT[cfr]="go through the center gate from the right and hover over the stuffed animal"; SIDE[cfr]=right; SCENE[cfr]=center; JS[cfr]=center_from_right
cd $RD
killport () { for p in $(pgrep -f "port $1\$"); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
roll_and_score () { # tag task
  local TAG=$1 C=$2
  rm -f $RUN/traj_${TAG}_*.npy
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=${SIDE[$C]} SCENE=${SCENE[$C]} NCH=14 APC=50 TRIALS=10 VIDEO=0 \
    PROMPT="${PROMPT[$C]}" TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}.log 2>&1
  { echo "== ablate_cmd_B: $TAG"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side ${JS[$C]}
    $TV $RD/gate_clearance.py --scene ${SCENE[$C]} --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
setsid $EV $PINENV XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.32 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY $RD/serve_head_only.py --ckpt $CK_PIN --norm $HFB/assets/gate_nav --port $HPORT \
  >> $RUN/sv_head_only.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do curl -s http://127.0.0.1:$HPORT/ >/dev/null && break; sleep 3; done
curl -s http://127.0.0.1:$HPORT/ >/dev/null || { echo "HEAD_TIMEOUT" >> $OUT; echo DONE > $RUN/ablate_cmd_B.done; exit 1; }
for CT in cfr:0.3 cfr:0.5 left:0.3 right:0.3 cfl:0.3; do
  C=${CT%:*}; T0=${CT#*:}; TAG=sdehead_t${T0/./}_$C; killport $PORT
  setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.38 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/serve_gate_sdedit.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav \
    --head-url http://127.0.0.1:$HPORT --pin-u $U --t0 $T0 --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; continue; }
  roll_and_score $TAG $C; killport $PORT
done
killport $HPORT
echo DONE > $RUN/ablate_cmd_B.done
