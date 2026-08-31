#!/bin/bash
# Eval both ablation arms after the scratch-sketch cells release the GPU (2026-08-30).
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
for k in $(seq 1 420); do [ -f $RUN/scrsketch.done ] && break; sleep 60; done
[ -f $RUN/scrsketch.done ] || { echo GATE_TIMEOUT > $RUN/ablation_evals.fail; exit 1; }
bash /home/dfliu/code/source-noise-mvp/scripts/run_synthonly_post.sh
bash /home/dfliu/code/source-noise-mvp/scripts/run_nosig_post.sh
CKA=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_synthonly/4999
CKB=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_nosig/4999
cd $RD
env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 \
  SNMVP_PIN_U=$RD/pin_U_mh16.npy SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 \
  SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=0 \
  /home/dfliu/code/openpi/.venv/bin/python synthpin_in_real.py --ckpt $CKA \
  --out $RUN/synthpin_synthonly.npz > $RUN/synthpin_synthonly.log 2>&1
env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 \
  SNMVP_PIN_U=$RD/pin_U_mh16.npy SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=0 \
  /home/dfliu/code/openpi/.venv/bin/python synthpin_in_real.py --ckpt $CKB \
  --out $RUN/synthpin_nosig.npz > $RUN/synthpin_nosig.log 2>&1
echo DONE > $RUN/ablation_evals.done
