#!/usr/bin/env bash
# GPU 0 only (GPU 1 = Denis's cosmos). (1) pin-source in-source parity check;
# (2) higher-budget few-shot on held-out task 21 (all demos, 3000 steps) for pin and
# scratch; (3) closed-loop eval on libero_object task 4. Tests whether a bigger
# few-shot budget lifts both arms off the floor before sweeping k.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
PIN=$CK/snmvp_src_pin/4999/params
SCR=$CK/snmvp_src_scratch/4999/params
UV=$HOME/.local/bin/uv
cd "$HOME/code/openpi"
rm -f "$RD/fewshot_v2.status"

# (1) pin-source parity: pin/4999 (unpinned serve) on an in-source object task (task 9)
bash "$RD/run_cl_eval.sh" "$CK/snmvp_src_pin/4999" NONE 9 8006 0 "$RD/cl_pinsrc_obj9.json" libero_object
echo "PARITY_DONE" >> "$RD/fewshot_v2.status"

adapt() {  # exp init pinU|"" episodes steps
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    SNMVP_INIT_CKPT="$2" SNMVP_PIN_U="$3" SNMVP_EPISODES="$4" \
    $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=$1 \
    --num-train-steps=$5 --save-interval=$5 --overwrite > "$RD/$1.log" 2>&1
  echo "ADAPT_$1=$?" >> "$RD/fewshot_v2.status"
}
adapt fs_pin_t21_full     "$PIN" "$RD/pin_U_pca_k5.npy" "$RD/fs_t21_full.json" 3000
adapt fs_scratch_t21_full "$SCR" ""                     "$RD/fs_t21_full.json" 3000

bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21_full/2999"     "$RD/prior_t21.npz" 4 8006 0 "$RD/cl_pin_t21_full.json"     libero_object
bash "$RD/run_cl_eval.sh" "$CK/fs_scratch_t21_full/2999" NONE               4 8006 0 "$RD/cl_scratch_t21_full.json" libero_object
echo "FEWSHOT_V2_DONE" >> "$RD/fewshot_v2.status"
