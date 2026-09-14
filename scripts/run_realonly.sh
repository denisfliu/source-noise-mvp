#!/bin/bash
# Real-only arms (2026-09-14, Denis: "try training with real only, scratch and ours, and see what happens"):
# the 100 real demonstrations (episodes 0-99, left/right only), no simulated data, no swap.
#   A. gate_scratch_real   plain pi0 fine-tune            -> run_sixcell_plain.sh scrreal
#   B. gate_pin_joint_realonly  pin + joint MDN head (xswap recipe minus swap) -> run_realonly_post.sh
#   C. real-anchor suite (B) + real-frame ledger (A, B) via real_vertical_probe.py
# Sequential on the one card; ~13 h. Sigma map for B is fitted on simulated frames like every arm (caveat).
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
SN=/home/dfliu/code/openpi-snmvp
rm -f $RUN/realonly.done $RUN/realonly_progress
[ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -ge 20 ] || { echo DISK_GUARD > $RUN/realonly.done; exit 1; }
gpu_wait () { for k in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0); [ "$u" -lt 2000 ] && return 0; sleep 30; done; return 1; }
cd $SN
# --- A: scratch, real only ---
gpu_wait
env -u VIRTUAL_ENV PYTHONPATH=$SN/src SNMVP_EPISODES=$RUN/eps_real.json XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY scripts/train.py pi0_gate3 --exp-name=gate_scratch_real --num-train-steps=5000 --lr-schedule.decay-steps=1000000 \
  --save-interval=5000 --seed=42 --no-wandb-enabled --overwrite > $RUN/arm_scrreal_train.log 2>&1
rm -rf $SN/checkpoints/pi0_gate3/gate_scratch_real/*/train_state
[ -d $SN/checkpoints/pi0_gate3/gate_scratch_real/4999/params ] || { echo A_TRAIN_FAILED > $RUN/realonly.done; exit 1; }
echo A_TRAINED >> $RUN/realonly_progress
bash /home/dfliu/code/source-noise-mvp/scripts/run_sixcell_plain.sh scrreal $SN/checkpoints/pi0_gate3/gate_scratch_real/4999 9082 10
echo A_EVAL >> $RUN/realonly_progress
# --- B: pin + head, real only ---
gpu_wait
env -u VIRTUAL_ENV PYTHONPATH=$SN/src SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$RD/pin_U_mh16.npy SNMVP_HEAD_DETACH=0 \
  SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 \
  SNMVP_EPISODES=$RUN/eps_real.json XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY scripts/train.py pi0_gate3 --exp-name=gate_pin_joint_realonly --num-train-steps=5000 --lr-schedule.decay-steps=1000000 \
  --save-interval=5000 --seed=42 --no-wandb-enabled --overwrite > $RUN/arm_realonly_train.log 2>&1
echo B_TRAINED >> $RUN/realonly_progress
bash /home/dfliu/code/source-noise-mvp/scripts/run_realonly_post.sh
echo B_POST >> $RUN/realonly_progress
# --- C: real-anchor suite + real-frame ledger ---
gpu_wait
cd $RD
CK=$SN/checkpoints/pi0_gate3/gate_pin_joint_realonly/4999
env -u VIRTUAL_ENV PYTHONPATH=$SN/src SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$RD/pin_U_mh16.npy SNMVP_HEAD_DETACH=0 \
  SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES=0 \
  $VENVPY synthpin_in_real.py --ckpt $CK --out $RUN/synthpin_realonly.npz > $RUN/synthpin_realonly.log 2>&1
PIN_TAG=realonly PIN_CK=$CK SIGMAP=$RD/sigma_map_realonly.json env -u VIRTUAL_ENV PYTHONPATH=$SN/src XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
  $VENVPY $RD/real_vertical_probe.py pin > $RUN/realism/probe_realonly_pin.log 2>&1
SCR_TAG=scrreal SCR_CK=$SN/checkpoints/pi0_gate3/gate_scratch_real/4999 env -u VIRTUAL_ENV PYTHONPATH=$SN/src XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
  $VENVPY $RD/real_vertical_probe.py scratch > $RUN/realism/probe_realonly_scratch.log 2>&1
echo DONE > $RUN/realonly.done
