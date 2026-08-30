#!/bin/bash
# Real-ablation arms (2026-08-30): (A) synthonly — gmsig recipe on synth episodes only
# (does real training data matter for real execution?); (B) nosig — gmsig recipe without
# sigma-conditioning (does the trust dial matter?). Each: train 5000 -> sigma-map where
# applicable -> six cells -> real-anchor suite.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
rm -f $RUN/ablations.done $RUN/ablations_progress
COMMON="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$RD/pin_U_mh16.npy SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0"
cd /home/dfliu/code/openpi-snmvp
# --- A: synth-only (sigma-conditioned like flagship, but no real episodes) ---
$COMMON SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 \
  SNMVP_EPISODES=$RUN/eps_synth.json \
  $VENVPY scripts/train.py pi0_gate3 --exp-name=gate_pin_joint_synthonly \
  --num-train-steps=5000 --lr-schedule.decay-steps=1000000 --save-interval=5000 \
  --seed=42 --no-wandb-enabled --overwrite > $RUN/arm_synthonly_train.log 2>&1
rm -rf checkpoints/pi0_gate3/gate_pin_joint_synthonly/*/train_state
echo A_TRAINED >> $RUN/ablations_progress
# --- B: no sigma-conditioning ---
$COMMON $VENVPY scripts/train.py pi0_gate3 --exp-name=gate_pin_joint_nosig \
  --num-train-steps=5000 --lr-schedule.decay-steps=1000000 --save-interval=5000 \
  --seed=42 --no-wandb-enabled --overwrite > $RUN/arm_nosig_train.log 2>&1
rm -rf checkpoints/pi0_gate3/gate_pin_joint_nosig/*/train_state
echo B_TRAINED >> $RUN/ablations_progress
echo DONE > $RUN/ablations.done
