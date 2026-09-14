#!/bin/bash
# Coarse-only cross-domain swap (2026-09-13, Denis: "plan like the simulator, fly like the pilot"): the xswap
# recipe (gmsig3 + matched-pair swap on real frames, p=0.5) but the swap replaces ONLY the pinned coordinates
# c = U^T a of the real chunk with the matched sim chunk's, keeping the pilot's residual; the transferred plan
# difference enters as the minimum-acceleration chunk with those band sums (data_loader._XDomSwap, mode=coarse).
# Seed 42. Post pipeline: scripts/run_xswapc_post.sh (readout gate, sigma map, six cells, real-anchor suite).
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
rm -f $RUN/arm_xswapc.done
[ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -ge 9 ] || { echo DISK_GUARD > $RUN/arm_xswapc.done; exit 1; }
for k in $(seq 1 40); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$u" -lt 2000 ] && break; sleep 15
done
cd /home/dfliu/code/openpi-snmvp
env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 \
  SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$RD/pin_U_mh16.npy SNMVP_HEAD_DETACH=0 \
  SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 \
  SNMVP_PIN_NOISE_COND=1 SNMVP_XDOM_SWAP=/home/dfliu/ctxrun/xswap_table.npz:0.5 \
  SNMVP_XDOM_SWAP_MODE=coarse SNMVP_XDOM_NORM=/home/dfliu/hf_bundle/gate-drone-pi0/assets/gate_nav/norm_stats.json \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0 \
  /home/dfliu/code/openpi/.venv/bin/python scripts/train.py pi0_gate3 \
  --exp-name=gate_pin_joint_xswapc --num-train-steps=5000 --lr-schedule.decay-steps=1000000 \
  --save-interval=5000 --seed=42 --no-wandb-enabled --overwrite \
  > $RUN/arm_xswapc_train.log 2>&1
