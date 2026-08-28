#!/bin/bash
# Real-in-the-loop emulator ladder (Denis direction, 2026-08-28): closed-loop sim physics
# with retrieved corpus observations. Arms: V0 retrieval-synth sanity; A1 head-on-real;
# A2 sim-twin commands (head on sim render, flow on real obs). gmsig3, L+R cells, 5 trials.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
U=$RD/pin_U_mh16.npy
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_gmsig3/4999
PORT=9095
SRV=serve_gate_pin_joint.py
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
rm -f $RUN/realloop.done $RUN/realloop_scores.txt
cd $RD
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
sleep 3
setsid $EV SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 \
  SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 \
  SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_gmsig3.json CLOG=$RUN/clog_realloop.npy \
  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY $RD/$SRV --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U \
  --port $PORT >> $RUN/sv_realloop.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT > $RUN/realloop.done; exit 1; }
run_arm () { # tag side extra_env...
  local TAG=$1 SIDE=$2; shift 2
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SIDE NCH=8 APC=50 TRIALS=5 VIDEO=0 \
    "$@" TRAJ=$RUN/traj_${TAG}_${SIDE}_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_${TAG}_${SIDE}.log 2>&1
  { echo "== realloop $TAG $SIDE"
    grep -a "retrieval" $RUN/roll_${TAG}_${SIDE}.log | tail -1
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
      --traj $RUN/traj_${TAG}_${SIDE}_*.npy --side $SIDE
    $TV $RD/gate_clearance.py --scene $SIDE --traj $RUN/traj_${TAG}_${SIDE}_*.npy
  } >> $RUN/realloop_scores.txt 2>&1
}
for SIDE in left right; do
  run_arm rlv0 $SIDE REALOBS=$RUN/obsidx_synth.npz
  run_arm rla1 $SIDE REALOBS=$RUN/obsidx_real.npz
  run_arm rla2 $SIDE REALOBS=$RUN/obsidx_real.npz SIMCMD=1
done
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
echo DONE > $RUN/realloop.done
