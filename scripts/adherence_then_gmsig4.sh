#!/bin/bash
# After the scratch chain releases the GPU: adherence baselines (gmsig3 + gmsigs7 on the SAME
# synth3 frames), then train gmsig4 = gmsig3 recipe + SNMVP_HEAD_COND_DROP=0.4,0.15,0.1 (text-adherence
# arm). Its post chain waits separately.
set -u
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
RUN=/home/dfliu/ctxrun
PYV=/home/dfliu/code/openpi/.venv/bin/python
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
HEADENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1"
for k in $(seq 1 720); do [ -f $RUN/ev6_scr3.done ] && break; sleep 60; done
cd $RD
$EV $HEADENV CUDA_VISIBLE_DEVICES=0 $PYV text_adherence_probe.py \
  --ckpt /home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_gmsig3/4999 \
  --pin-u $RD/pin_U_mh16.npy --data-dir data_gate_synth3 --save adh_gmsig3.npz \
  > $RUN/adh_gmsig3.log 2>&1
$EV $HEADENV CUDA_VISIBLE_DEVICES=0 $PYV text_adherence_probe.py \
  --ckpt /home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate/gate_pin_joint_gmsigs7/4999 \
  --pin-u $RD/pin_U_mh16.npy --data-dir data_gate_synth3 --save adh_gmsigs7.npz \
  > $RUN/adh_gmsigs7.log 2>&1
cd /home/dfliu/code/openpi-snmvp
env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src $HEADENV \
  SNMVP_PIN_U=$RD/pin_U_mh16.npy SNMVP_HEAD_COND_DROP=0.4,0.15,0.1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0 \
  /home/dfliu/code/openpi/.venv/bin/python scripts/train.py pi0_gate3 \
  --exp-name=gate_pin_joint_gmsig4 --num-train-steps=5000 --lr-schedule.decay-steps=1000000 \
  --save-interval=5000 --seed=42 --no-wandb-enabled --overwrite \
  > $RUN/arm_gmsig4_train.log 2>&1
echo ADH_GMSIG4_TRAIN_DONE > $RUN/adh_gmsig4.done
