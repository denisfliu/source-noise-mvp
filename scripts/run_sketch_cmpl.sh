#!/bin/bash
# Corrective-sketch screen (Denis go, 2026-08-25): gmsig3 (flagship, no CFG) served with
# SNMVP_PIN_PROMPT=sketch_cmpl.json — the sketch owns the switch segment at sigma=0, the head
# owns everything else (trust dial + hysteresis unchanged). CMPL 5 trials, screen tier
# VIDEO=0. Judge is route-clean (wrong-direction aperture passes fail).
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_mh16.npy
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
HEADENV="SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_gmsig3.json"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_gmsig3/4999
SRV=serve_gate_pin_joint.py
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
PORT=9060
rm -f $RUN/sketch_cmpl.done $RUN/traj_skcmpl_*.npy $RUN/sketch_cmpl_scores.txt
cd $RD
$VENVPY $RD/make_sketch_cmpl.py
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
sleep 3
setsid $EV $BASE $HEADENV SNMVP_PIN_PROMPT=$RD/sketch_cmpl.json CLOG=$RUN/clog_skcmpl.npy \
  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY $RD/$SRV --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U \
  --port $PORT >> $RUN/sv_skcmpl.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT >> $RUN/sketch_cmpl.done; exit 1; }
env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=left SCENE=left_and_center NCH=14 APC=50 TRIALS=5 VIDEO=0 \
  PROMPT="$CMP_L" TRAJ=$RUN/traj_skcmpl_{t}.npy $TV $RD/gate_rollout_batch.py \
  > $RUN/roll_skcmpl.log 2>&1
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
{ echo "== gmsig3 + corrective sketch, cmpl (route-clean judge)"
  $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
    --traj $RUN/traj_skcmpl_*.npy --side left_and_center
  $TV $RD/gate_clearance.py --scene left_and_center --traj $RUN/traj_skcmpl_*.npy
} >> $RUN/sketch_cmpl_scores.txt 2>&1
echo DONE >> $RUN/sketch_cmpl.done
