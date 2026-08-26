#!/bin/bash
# Re-fly ONLY compound-right for the ctl checkpoint: the original cell OOM'd (two compound splat
# clients + server > 24 GB, fixed to sequential in the chain scripts). Waits for the gmm arm's
# chain to release the GPU, then serves ctl and flies the one missing cell (5-trial screen tier).
# Scores append to ev6_ctl_ctr_scores.txt so the ctl row lives in one place; marker cmpr_ctl.done.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_gate_rrr_k5.npy
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
EXTRA="SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate/gate_pin_joint_ctl/4999
PORT=8936
SRV=serve_gate_pin_joint.py
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
rm -f $RUN/cmpr_ctl.done
for k in $(seq 1 1440); do [ -f $RUN/arm_gmm.done ] && break; sleep 60; done
[ -f $RUN/arm_gmm.done ] || { echo WAIT_TIMEOUT > $RUN/cmpr_ctl.done; exit 1; }
for k in $(seq 1 240); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$u" -lt 2000 ] && break; sleep 60
done
cd $RD
rm -f $RUN/traj_ctl_cmpr_*.npy
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
sleep 3
setsid $EV $BASE $EXTRA CLOG=$RUN/clog_ctl_cmpr.npy XLA_PYTHON_CLIENT_PREALLOCATE=true \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 $VENVPY $RD/$SRV \
  --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT \
  >> $RUN/sv_cmpr_ctl.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT > $RUN/cmpr_ctl.done; exit 1; }
env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right_and_center NCH=14 APC=50 TRIALS=5 \
  VIDEO=0 PROMPT="$CMP_R" TRAJ=$RUN/traj_ctl_cmpr_{t}.npy \
  $TV $RD/gate_rollout_batch.py > $RUN/roll_ctl_cmpr.log 2>&1
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
{ echo "== ctl cmpr REFLY (judge: right_and_center)"
  $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
    --traj $RUN/traj_ctl_cmpr_*.npy --side right_and_center
  $TV $RD/gate_clearance.py --scene right_and_center --traj $RUN/traj_ctl_cmpr_*.npy
} >> $RUN/ev6_ctl_ctr_scores.txt 2>&1
echo DONE > $RUN/cmpr_ctl.done
