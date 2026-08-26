#!/bin/bash
# Source LoRA training for the VLA few-shot study: pin (top-5 PCA subspace) then
# scratch, sequential on GPU 1 (GPU 0 reserved). Holds out the 8 few-shot tasks via
# SNMVP_EPISODES. Short (5000 steps each). Checkpoints saved for few-shot adaptation.
set -u
cd ~/code/openpi
UV=~/.local/bin/uv
LOGD=~/code/source-noise-mvp/experiments/rung3
EP=$LOGD/source_episodes.json
rm -f $LOGD/src_train.status
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 SNMVP_EPISODES=$EP
echo "[src] start $(date -u +%H:%M:%S)" >> $LOGD/src_train.status

SNMVP_PIN_U=$LOGD/pin_U_pca_k5.npy $UV run scripts/train.py pi0_libero_low_mem_finetune \
  --exp-name=snmvp_src_pin --num-train-steps=5000 --save-interval=2500 --overwrite > $LOGD/src_pin.log 2>&1
echo "PIN_DONE=$? $(date -u +%H:%M:%S)" >> $LOGD/src_train.status

$UV run scripts/train.py pi0_libero_low_mem_finetune \
  --exp-name=snmvp_src_scratch --num-train-steps=5000 --save-interval=2500 --overwrite > $LOGD/src_scratch.log 2>&1
echo "SCRATCH_DONE=$? $(date -u +%H:%M:%S)" >> $LOGD/src_train.status
echo "SRC_TRAIN_ALL_DONE" >> $LOGD/src_train.status
