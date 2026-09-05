#!/bin/bash
# Re-fly the five sketched-compound rows on the FLAGSHIP (xswap) checkpoints, both training
# seeds (2026-09-02). Every earlier sketch row was flown on gmsig3/gmsig3s7; the paper
# attributes them to the full method, so re-attribute honestly. Settings identical to the
# original rows (sketch JSONs unchanged: carrot=0, sigma per JSON, APC=50, 5 trials/cell).
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_mh16.npy
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
HEADENV="SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1"
SRV=serve_gate_pin_joint.py
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
PORT=9130
OUT=$RUN/xsk_scores.txt
rm -f $RUN/xsk.done $OUT
cd $RD
cell () { # tag ckpt sigmap sketch side scene prompt
  local TAG=$1 CK=$2 SM=$3 SK=$4 SIDE=$5 SCENE=$6 PROMPT=$7
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV $BASE $HEADENV SNMVP_SIGMA_MAP=$RD/$SM SNMVP_PIN_PROMPT=$RD/$SK \
    CLOG=$RUN/clog_${TAG}.npy XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/$SRV --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U \
    --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    PROMPT="$PROMPT" TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_${TAG}.log 2>&1
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { echo "== xswap sketch re-attribution: $TAG (sketch=$SK)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
      --traj $RUN/traj_${TAG}_*.npy --side $SCENE
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
  } >> $OUT 2>&1
}
for S in 42 s7; do
  if [ $S = 42 ]; then CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999; SM=sigma_map_xswap.json; P=xsk42
  else CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswaps7/4999; SM=sigma_map_xswaps7.json; P=xsks7; fi
  cell ${P}_cmpl_denis $CK $SM sketch_cmpl_denis.json    left  left_and_center  "$CMP_L"
  cell ${P}_cmpr_r1    $CK $SM sketch_cmpr_denis_r1.json right right_and_center "$CMP_R"
  cell ${P}_cmpl_min4  $CK $SM sketch_cmpl_min4.json     left  left_and_center  "$CMP_L"
  cell ${P}_cmpl_min4s $CK $SM sketch_cmpl_min4s.json    left  left_and_center  "$CMP_L"
  cell ${P}_cmpr_min5f $CK $SM sketch_cmpr_min5f.json    right right_and_center "$CMP_R"
done
echo DONE > $RUN/xsk.done
