#!/usr/bin/env bash
# PCA-component sweep for the pin: does raising the pinned-subspace dimension K raise the
# few-shot ceiling or just move/steepen the overtraining decline? One K per invocation.
# Full pipeline: source-pin pretrain (5000, same recipe as snmvp_src_pin) -> few-shot
# adapt on task 21 (45 demos, step-checkpointed) -> K-dim state->c prior -> eval sweep.
# Deletes the adapt step checkpoints after eval (8.7G each) to stay disk-safe.
# Args: K GPU PORT [WAITFILE WAITTOKEN]  (optional wait frees a shared GPU first)
set -u
K=$1; GPU=$2; PORT=$3; WAITFILE=${4:-}; WAITTOKEN=${5:-}
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune
UV=$HOME/.local/bin/uv
U=$RD/pin_U_pca_k${K}.npy
ST=$RD/pca_k${K}.status
cd "$HOME/code/openpi"
rm -f "$ST"

if [ -n "$WAITFILE" ]; then
  for i in $(seq 1 1200); do grep -q "$WAITTOKEN" "$WAITFILE" 2>/dev/null && break; sleep 20; done
  echo "WAIT_DONE $(date -u +%H:%M:%S)" >> "$ST"
fi

# 1) source-pin pretrain (mirror snmvp_src_pin: 5000 steps, save 2500 -> /4999)
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  SNMVP_PIN_U="$U" SNMVP_EPISODES="$RD/source_episodes.json" \
  $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=snmvp_src_pin_k${K} \
  --num-train-steps=5000 --save-interval=2500 --overwrite > "$RD/src_pin_k${K}.log" 2>&1
echo "SRC_DONE=$? $(date -u +%H:%M:%S)" >> "$ST"

# 2) few-shot adapt on task 21, step-checkpointed
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  SNMVP_INIT_CKPT="$CK/snmvp_src_pin_k${K}/4999/params" SNMVP_PIN_U="$U" SNMVP_EPISODES="$RD/fs_t21_full.json" \
  $UV run scripts/train.py pi0_libero_low_mem_finetune --exp-name=fs_pin_t21_k${K} \
  --num-train-steps=3000 --save-interval=500 --keep-period=500 --overwrite > "$RD/fs_pin_t21_k${K}.log" 2>&1
echo "ADAPT_DONE=$? $(date -u +%H:%M:%S)" >> "$ST"

# 3) K-dim state->c prior on the 45 demos (CPU)
SNMVP_PIN_U="$U" SNMVP_EPISODES="$RD/fs_t21_full.json" SNMVP_PRIOR_OUT="$RD/prior_t21_k${K}.npz" \
  SNMVP_NB=60 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
  $UV run python make_prior.py > "$RD/prior_t21_k${K}.log" 2>&1
echo "PRIOR_DONE $(grep -o 'R\^2 [0-9.]*' "$RD/prior_t21_k${K}.log" | tail -1)" >> "$ST"

# 4) eval sweep (30 trials/point), then delete the step checkpoint to free disk
for S in 1500 2000 2500 2999; do
  if [ -d "$CK/fs_pin_t21_k${K}/$S" ]; then
    bash "$RD/run_cl_eval.sh" "$CK/fs_pin_t21_k${K}/$S" "$RD/prior_t21_k${K}.npz" 4 "$PORT" "$GPU" \
      "$RD/cl_pin_t21_k${K}_s${S}.json" libero_object 30
    rm -rf "$CK/fs_pin_t21_k${K}/$S"
  fi
  echo "STEP_$S done" >> "$ST"
done
echo "PCA_K${K}_DONE $(date -u +%H:%M:%S)" >> "$ST"
