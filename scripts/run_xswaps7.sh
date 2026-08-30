#!/bin/bash
# S3 cross-supervised training (Denis, 2026-08-28): gmsig3 recipe + matched-pair chunk swap
# on real frames (SNMVP_XDOM_SWAP p=0.5). Mixed training throughout — no phases, no drift.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
rm -f $RUN/arm_xswaps7.done
:
[ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -ge 9 ] || { echo DISK_GUARD > $RUN/arm_xswaps7.done; exit 1; }
for k in $(seq 1 40); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$u" -lt 2000 ] && break; sleep 15
done
cd /home/dfliu/code/openpi-snmvp
env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 \
  SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$RD/pin_U_mh16.npy SNMVP_HEAD_DETACH=0 \
  SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 \
  SNMVP_PIN_NOISE_COND=1 SNMVP_XDOM_SWAP=/home/dfliu/ctxrun/xswap_table.npz:0.5 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0 \
  /home/dfliu/code/openpi/.venv/bin/python scripts/train.py pi0_gate3 \
  --exp-name=gate_pin_joint_xswaps7 --num-train-steps=5000 --lr-schedule.decay-steps=1000000 \
  --save-interval=5000 --seed=7 --no-wandb-enabled --overwrite \
  > $RUN/arm_xswaps7_train.log 2>&1
bash /home/dfliu/code/source-noise-mvp/scripts/run_xswaps7_post.sh
# real-anchor suite on the xswap checkpoint
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswaps7/4999
cd $RD
env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 \
  SNMVP_PIN_U=$RD/pin_U_mh16.npy SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 \
  SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=0 \
  /home/dfliu/code/openpi/.venv/bin/python synthpin_in_real.py --ckpt $CK \
  --out $RUN/synthpin_xswaps7.npz > $RUN/synthpin_xswaps7.log 2>&1
echo ALLDONE >> $RUN/arm_xswaps7.done
