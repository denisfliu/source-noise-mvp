#!/bin/bash
# A/B of the joint pin server through the real serving path: fused (SNMVP_FUSED=1) vs two-pass (0),
# same checkpoint, same real episode replayed by the dry flight client, same server RNG seed.
set -u
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999
U=$RD/pin_U_mh16.npy; PORT=9250; OUT=/home/dfliu/ctxrun/realism
EP=$RD/data_gate_real/ep_0061.npz
for FUSED in ${FUSED_LIST:-1 0}; do
  for p in $(pgrep -f "serve_gate_pin_joint.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done; sleep 2
  setsid env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
    SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 \
    SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_xswap.json \
    SNMVP_FUSED=$FUSED SNMVP_SERVE_SEED=1234 CLOG=$OUT/clog_ab_fused${FUSED}${SUFFIX:-}.npy \
    $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT \
    > $OUT/sv_ab_fused$FUSED.log 2>&1 < /dev/null & disown
  for k in $(seq 1 120); do ss -ltn | grep -q ":$PORT " && break; sleep 2; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT fused=$FUSED"; continue; }
  cd /home/dfliu/code/dronevla2.0
  echo "===== FUSED=$FUSED (warm-up pass, then timed pass)"
  $VENVPY tools/gate_dry_client.py --host 127.0.0.1 --port $PORT --task right --episode $EP --replans 3 > /dev/null 2>&1
  $VENVPY tools/gate_dry_client.py --host 127.0.0.1 --port $PORT --task right --episode $EP --replans 8 2>&1 | grep -v "^INFO" | tee $OUT/dry_ab_fused${FUSED}${SUFFIX:-}.txt
  for p in $(pgrep -f "serve_gate_pin_joint.p[y] .*port $PORT"); do kill -9 "$p" 2>/dev/null; done; sleep 2
done
