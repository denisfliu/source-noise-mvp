#!/bin/bash
# Sigma-gated serve test on the gmmmh checkpoint (2026-08-21): re-fly the three failing cells
# (right, CFL, CFR; 10 trials each, VIDEO=0 screen tier) with SNMVP_GMM_SIGGATE softening the
# pin toward plain denoising when the head's own sigma* is high. Thresholds = demo-sigma
# quantiles p60/p90 (3.74/9.72), amin=0.25. Direct comparison against the ungated row
# (arm_gmmmh/ctr_gmmmh scores). CLOG rows carry [pos, c, pi, sigma*, alpha].
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_mh16.npy
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
EXTRA="SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_GMM_SIGGATE=3.74,9.72,0.25"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate/gate_pin_joint_gmmmh/4999
PORT=8980
SRV=serve_gate_pin_joint.py
CFL="go through the center gate from the left and hover over the stuffed animal"
CFR="go through the center gate from the right and hover over the stuffed animal"
rm -f $RUN/sg_gmmmh.done $RUN/sg_gmmmh_scores.txt $RUN/traj_sggmmmh_*.npy
for k in $(seq 1 240); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$u" -lt 2000 ] && break; sleep 60
done
cd $RD
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
sleep 3
setsid $EV $BASE $EXTRA CLOG=$RUN/clog_sggmmmh.npy XLA_PYTHON_CLIENT_PREALLOCATE=true \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 $VENVPY $RD/$SRV \
  --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT \
  >> $RUN/sv_sggmmmh.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT > $RUN/sg_gmmmh.done; exit 1; }
roll () {  # tag side scene prompt nch
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$2 SCENE=$3 NCH=$5 APC=50 TRIALS=10 VIDEO=0 \
    PROMPT="$4" TRAJ=$RUN/traj_sggmmmh_$1_{t}.npy \
    $TV $RD/gate_rollout_batch.py > $RUN/roll_sggmmmh_$1.log 2>&1
}
roll right right right "go through the gate on the right and hover over the stuffed animal" 8
roll cfl left  center "$CFL" 10
roll cfr right center "$CFR" 10
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
{ for spec in "right right right" "cfl center_from_left center" "cfr center_from_right center"; do
    set -- $spec
    echo "== sg_gmmmh $1 (judge: $2)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
      --traj $RUN/traj_sggmmmh_$1_*.npy --side $2
    $TV $RD/gate_clearance.py --scene $3 --traj $RUN/traj_sggmmmh_$1_*.npy
  done } > $RUN/sg_gmmmh_scores.txt 2>&1
grep -qa "clearance-clean" $RUN/sg_gmmmh_scores.txt && echo DONE > $RUN/sg_gmmmh.done \
  || echo SCORE_FAILED > $RUN/sg_gmmmh.done
