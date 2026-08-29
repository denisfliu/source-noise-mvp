#!/bin/bash
# Carrot/kick divergence test (Denis, 2026-08-28): hand-drawn CMPL sketch, 0.4 m lateral
# kick mid-corridor (step 100). K0 no-carrot+kick / K1 carrot20+kick / K2 carrot20 alone.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
U=$RD/pin_U_mh16.npy
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_gmsig3/4999
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
PORT=9100
rm -f $RUN/carrot.done $RUN/carrot_scores.txt
cd $RD
cell () { # tag sketch kick
  local TAG=$1 SK=$2 KK=$3
  for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 \
    SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 \
    SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_gmsig3.json SNMVP_PIN_PROMPT=$RD/$SK \
    CLOG=$RUN/clog_${TAG}.npy XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
    CUDA_VISIBLE_DEVICES=0 $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate \
    --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=left SCENE=left_and_center NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    ${KK:+KICK=$KK} PROMPT="$CMP_L" TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_${TAG}.log 2>&1
  for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { echo "== carrot test $TAG (sketch=$SK kick=${KK:-none})"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
      --traj $RUN/traj_${TAG}_*.npy --side left_and_center
    $TV $RD/gate_clearance.py --scene left_and_center --traj $RUN/traj_${TAG}_*.npy
  } >> $RUN/carrot_scores.txt 2>&1
}
cell ck0 sketch_cmpl_denis.json "100:0,-0.4,0"
cell ck1 sketch_cmpl_denis_c20.json "100:0,-0.4,0"
cell ck2 sketch_cmpl_denis_c20.json ""
echo DONE > $RUN/carrot.done
