#!/bin/bash
# Domain-split training experiment (Denis go, 2026-08-27): synth teaches the pin (phase A,
# synth-only episodes, full recipe), real teaches the denoising (phase B, resume, real-only,
# HEAD_LAM=0 — flow matching with the pin carried, head frozen). Then sigma map + six cells.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
NAME=dsplit
CKDIR=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_$NAME
BASEENV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$RD/pin_U_mh16.npy SNMVP_HEAD_DETACH=0 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0"
rm -f $RUN/dsplit.done
# wait for the probe chain to release the GPU
for k in $(seq 1 120); do [ -f $RUN/xdom_probes.done ] && break; sleep 60; done
[ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -ge 9 ] || { echo DISK_GUARD > $RUN/dsplit.done; exit 1; }
cd /home/dfliu/code/openpi-snmvp
# Phase A: synth-only, head on
$BASEENV SNMVP_HEAD_LAM=0.3 SNMVP_EPISODES=$RUN/eps_synth.json \
  $VENVPY scripts/train.py pi0_gate3 --exp-name=gate_pin_joint_$NAME \
  --num-train-steps=4000 --lr-schedule.decay-steps=1000000 --save-interval=4000 \
  --seed=42 --no-wandb-enabled --overwrite > $RUN/arm_${NAME}_trainA.log 2>&1
[ -d "$CKDIR/3999/params" ] || { echo PHASE_A_NO_CKPT > $RUN/dsplit.done; exit 1; }
echo PHASE_A_DONE >> $RUN/dsplit_progress
# Phase B: real-only, head loss off, resume
$BASEENV SNMVP_HEAD_LAM=0.0 SNMVP_EPISODES=$RUN/eps_real.json \
  $VENVPY scripts/train.py pi0_gate3 --exp-name=gate_pin_joint_$NAME \
  --num-train-steps=5500 --lr-schedule.decay-steps=1000000 --save-interval=5500 \
  --seed=42 --no-wandb-enabled --resume > $RUN/arm_${NAME}_trainB.log 2>&1
[ -d "$CKDIR/5499/params" ] || { echo PHASE_B_NO_CKPT > $RUN/dsplit.done; exit 1; }
rm -rf $CKDIR/*/train_state
echo PHASE_B_DONE >> $RUN/dsplit_progress
echo DONE > $RUN/dsplit.done
