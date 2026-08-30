#!/bin/bash
# Overnight pin applications (2026-08-29, Denis asleep): tempo verb (same route at
# 0.6x/1.0x/1.5x via sketch step_m), orbit primitive (1.5 loops around the gate — no demo
# ever orbits), figure-8 in open space. xswap seed-42 checkpoint, sigma=0, carrot on.
# Gated on the xswaps7 replication chain finishing.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
U=$RD/pin_U_mh16.npy
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999
PORT=9120
rm -f $RUN/pinapps.done $RUN/pinapps_scores.txt
for k in $(seq 1 400); do [ -f $RUN/arm_xswaps7.done ] && break; sleep 60; done
cd $RD
for SK in tempo06 tempo10 tempo15 orbit fig8; do
  for p in $(pgrep -f "serve_gate_pin_joint.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 \
    SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 \
    SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 \
    SNMVP_SIGMA_MAP=$RD/sigma_map_xswap.json SNMVP_PIN_PROMPT=$RD/sketch_$SK.json \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav \
    --pin-u $U --port $PORT >> $RUN/sv_app_$SK.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    TRAJ=$RUN/traj_app_${SK}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_app_$SK.log 2>&1
  for p in $(pgrep -f "serve_gate_pin_joint.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { echo "== pin app $SK"
    env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu \
      CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_app_${SK}_*.npy --side right
    $TV $RD/gate_clearance.py --scene right --traj $RUN/traj_app_${SK}_*.npy
  } >> $RUN/pinapps_scores.txt 2>&1
done
echo DONE > $RUN/pinapps.done
