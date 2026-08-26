#!/usr/bin/env bash
# Few-shot adapt (pin + scratch) on the easy held-out libero_object tasks 21 and 28.
# GPU 0 does task 21 (pin then scratch), GPU 1 does task 28. Init from the source models.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
PIN=$CK/snmvp_src_pin/4999/params
SCR=$CK/snmvp_src_scratch/4999/params
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/fs_easy.status"

adapt() {  # gpu exp init_ckpt pinU|"" episodes  (empty pinU => pin off)
  CUDA_VISIBLE_DEVICES=$1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    SNMVP_INIT_CKPT="$3" SNMVP_PIN_U="$4" SNMVP_EPISODES="$5" \
    $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=$2 \
    --num-train-steps=800 --save-interval=800 --overwrite > "$RD/$2.log" 2>&1
}

( adapt 0 fs_pin_t21     "$PIN" "$RD/pin_U_pca_k5.npy" "$RD/fs_t21_k10.json"
  adapt 0 fs_scratch_t21 "$SCR" ""                     "$RD/fs_t21_k10.json"
  echo "G0_DONE" >> "$RD/fs_easy.status" ) &
( adapt 1 fs_pin_t28     "$PIN" "$RD/pin_U_pca_k5.npy" "$RD/fs_t28_k10.json"
  adapt 1 fs_scratch_t28 "$SCR" ""                     "$RD/fs_t28_k10.json"
  echo "G1_DONE" >> "$RD/fs_easy.status" ) &
wait
echo "FS_EASY_DONE" >> "$RD/fs_easy.status"
