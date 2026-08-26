#!/bin/bash
# CFG language-sharpener compound screen (Denis go, 2026-08-25): gmsig4 served with
# SNMVP_GMM_LANG_CFG in {2,4}, compounds cmpl+cmpr 5 trials each (screen tier, VIDEO=0),
# trust dial + hysteresis unchanged. Scores per w: cfg${W}_gmsig4_scores.txt.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_mh16.npy
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
HEADENV="SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_gmsig4.json"
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_gmsig4/4999
SRV=serve_gate_pin_joint.py
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
rm -f $RUN/cfgscreen.done
cd $RD
for W in 2 4; do
  PORT=$((9050 + W))
  rm -f $RUN/traj_cfg${W}g4_*.npy $RUN/cfg${W}_gmsig4_scores.txt
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV $BASE $HEADENV SNMVP_GMM_LANG_CFG=$W CLOG=$RUN/clog_cfg${W}g4.npy \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/$SRV --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U \
    --port $PORT >> $RUN/sv_cfg${W}g4.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT w=$W" >> $RUN/cfgscreen.done; continue; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=left SCENE=left_and_center NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    PROMPT="$CMP_L" TRAJ=$RUN/traj_cfg${W}g4_cmpl_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_cfg${W}g4_cmpl.log 2>&1
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right_and_center NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    PROMPT="$CMP_R" TRAJ=$RUN/traj_cfg${W}g4_cmpr_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_cfg${W}g4_cmpr.log 2>&1
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { for spec in "cmpl left_and_center" "cmpr right_and_center"; do
      set -- $spec
      echo "== gmsig4 CFG w=$W $1 (judge: $2)"
      $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
        --traj $RUN/traj_cfg${W}g4_$1_*.npy --side $2
      $TV $RD/gate_clearance.py --scene $2 --traj $RUN/traj_cfg${W}g4_$1_*.npy
    done } >> $RUN/cfg${W}_gmsig4_scores.txt 2>&1
done
echo DONE >> $RUN/cfgscreen.done
