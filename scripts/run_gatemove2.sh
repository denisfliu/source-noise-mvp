#!/bin/bash
# Arbitrary-gate-pose sweep (Denis, 2026-08-28): the right gate moved/rotated in the splat
# (visals+geometry), auto 4-point sketch through each new pose (carrot=20, sigma=0), xswap
# checkpoint. 7 poses x 5 trials; scored by moved_gate_cell.py vs the transformed aperture.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
U=$RD/pin_U_mh16.npy
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
PORT=9104
rm -f $RUN/gatemove2.done $RUN/gatemove2_scores.txt

cd $RD
POSES="100,-1.26,1.50 180,0,0 -35,1.34,0.75 90,0.74,1.95 90,-0.26,0.85 -45,0,0 0,0.5,-0.3"
for TF in $POSES; do
  TAG=mx$(echo $TF | tr ',.-' '__m')
  IFS=, read DY DX DYY <<< "$TF"
  $VENVPY $RD/moved_gate_cell.py --make --dyaw $DY --dx $DX --dy $DYY --tag $TAG >> $RUN/gatemove2_scores.txt
  for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 \
    SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 \
    SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_xswap.json SNMVP_PIN_PROMPT=$RD/sketch_mg_$TAG.json \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav \
    --pin-u $U --port $PORT >> $RUN/sv_$TAG.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    GATE_TF=$TF TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_$TAG.log 2>&1
  for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  $VENVPY $RD/moved_gate_cell.py --score --dyaw $DY --dx $DX --dy $DYY --tag $TAG \
    --traj $RUN/traj_${TAG}_*.npy >> $RUN/gatemove2_scores.txt 2>&1
done
echo DONE > $RUN/gatemove2.done
