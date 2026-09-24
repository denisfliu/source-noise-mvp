#!/bin/bash
# Arbitrary-maneuver test on the REAL-ONLY pin (2026-09-15): orbit + figure-8 sketches through gate_pin_joint_realonly,
# same pipeline as scripts/run_pin_apps.sh (xswap). 5 trials each, right scene, sigma per sketch json (0), carrot 20.
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
U=$RD/pin_U_mh16.npy; CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_realonly/4999; PORT=9121
OUT=$RUN/realonly_apps_scores.txt; rm -f $OUT $RUN/realonly_apps.done
cd $RD
for SK in orbit fig8; do
  for p in $(pgrep -f "serve_gate_pin_joint.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done; sleep 3
  setsid env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U \
    SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 \
    SNMVP_SIGMA_MAP=$RD/sigma_map_realonly.json SNMVP_PIN_PROMPT=$RD/sketch_$SK.json \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_app_realonly_$SK.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $SK" >> $OUT; continue; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=right SCENE=right NCH=14 APC=50 TRIALS=${TRIALS:-5} TRIAL0=${TRIAL0:-1} VIDEO=0 \
    TRAJ=$RUN/traj_app_realonly_${SK}_{t}.npy $TV $RD/gate_rollout_batch.py > $RUN/roll_app_realonly_$SK.log 2>&1
  for p in $(pgrep -f "serve_gate_pin_joint.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { echo "== real-only pin app $SK"
    env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_app_realonly_${SK}_*.npy --side right
    $TV $RD/gate_clearance.py --scene right --traj $RUN/traj_app_realonly_${SK}_*.npy
  } >> $OUT 2>&1
done
echo DONE > $RUN/realonly_apps.done
