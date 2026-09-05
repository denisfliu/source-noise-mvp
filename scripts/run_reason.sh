#!/bin/bash
# VLM movement reasoner as the coarse command source (2026-09-03). Qwen2.5-VL-3B service + xswap pin
# server (SNMVP_PIN_REASON); the reasoner's words fill the coarse xyz coordinates, the head keeps
# the rest. Cells: the four atomics (5 trials) then the L->C compound (5 trials).
#   bash scripts/run_reason.sh [mode=coarse_xyz] [cells: left right cfl cfr cmpl]
set -u
MODE=${1:-coarse_xyz}; shift || true
CELLS=${@:-left right cfl cfr cmpl}
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
VLMPY=/home/dfliu/code/vlmenv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_xswap.json"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999
VPORT=9190; PORT=9180
OUT=$RUN/reason_scores.txt
rm -f $RUN/reason.done
declare -A PROMPT SIDE SCENE
PROMPT[left]="go through the gate on the left and hover over the stuffed animal";   SIDE[left]=left;  SCENE[left]=left
PROMPT[right]="go through the gate on the right and hover over the stuffed animal"; SIDE[right]=right; SCENE[right]=right
PROMPT[cfl]="go through the center gate from the left and hover over the stuffed animal";  SIDE[cfl]=left;  SCENE[cfl]=center
PROMPT[cfr]="go through the center gate from the right and hover over the stuffed animal"; SIDE[cfr]=right; SCENE[cfr]=center
PROMPT[cmpl]="go through the gate on the left, then through the center gate and hover over the stuffed animal"; SIDE[cmpl]=left; SCENE[cmpl]=left_and_center
JUDGE_SIDE() { case $1 in left|right) echo $1;; cfl) echo center_from_left;; cfr) echo center_from_right;; cmpl) echo left_and_center;; esac; }
cd $RD
# VLM service (once)
if ! curl -s http://127.0.0.1:$VPORT/health >/dev/null; then
  setsid env CUDA_VISIBLE_DEVICES=0 $VLMPY $RD/vlm_reason_server.py --port $VPORT >> $RUN/sv_vlm_reason.log 2>&1 </dev/null & disown
  for k in $(seq 1 120); do curl -s http://127.0.0.1:$VPORT/health >/dev/null && break; sleep 5; done
  curl -s http://127.0.0.1:$VPORT/health >/dev/null || { echo "VLM_TIMEOUT" >> $OUT; exit 1; }
fi
for C in $CELLS; do
  TAG=rs_${MODE}_$C
  for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done; sleep 3
  rm -f $RUN/reason_${TAG}.jsonl
  setsid $EV $PINENV SNMVP_PIN_REASON=http://127.0.0.1:$VPORT SNMVP_REASON_MODE=$MODE SNMVP_REASON_LOG=$RUN/reason_${TAG}.jsonl \
    CLOG=$RUN/clog_${TAG}.npy $GPU $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate \
    --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; continue; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=${SIDE[$C]} SCENE=${SCENE[$C]} NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    PROMPT="${PROMPT[$C]}" TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_${TAG}.log 2>&1
  for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { echo "== reason: $TAG (VLM coarse words, mode=$MODE, task=$C)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side $(JUDGE_SIDE $C)
    $TV $RD/gate_clearance.py --scene ${SCENE[$C]} --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
done
echo DONE > $RUN/reason.done
