#!/usr/bin/env bash
# Adapted-protocol HEAD-TO-HEAD: does the RRR pin help few-shot CLOSED-LOOP vs a scratch few-shot
# baseline, on a held-out goal task (language-driven) and object task (state-driven)? Self-contained
# and SSH-drop safe: launch with
#   setsid bash run_headtohead.sh </dev/null >htoh_driver.log 2>&1 & disown
# Steps: (0) scratch source if missing; per task (goal 11, object 21): build fs episodes; adapt
# pin arm (from RRR source-pin, pin active) and scratch arm (from scratch source, no pin); fit the
# pin's state->c prior; closed-loop eval pin (prior) vs scratch (unpinned). All -> htoh_vlm.status.
set -uo pipefail
RD=$HOME/code/source-noise-mvp/experiments/rung3
CKD=$HOME/code/openpi/checkpoints/pi0_libero_shared
U=$RD/pin_U_vlm_ctx_k5_shared.npy
GPU=${SNMVP_GPU:-0}; TRIALS=${1:-15}; STEPS=${2:-2000}
UV=$HOME/.local/bin/uv
LVENV=$HOME/code/openpi/examples/libero/.venv/bin/python
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
export PYTHONPATH="$HOME/code/openpi/third_party/libero"
export LIBERO_CONFIG_PATH="$HOME/code/libero-config"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU
cd "$HOME/code/openpi"
ST=$RD/htoh_vlm.status
echo "START $(date -u +%H:%M:%S) trials=$TRIALS steps=$STEPS" >> "$ST"

train() {  # exp init pinU|"" episodes  (assignment-from-expansion is not honored, so export/unset)
  if [ -n "$3" ]; then export SNMVP_PIN_U="$3"; else unset SNMVP_PIN_U; fi
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
    SNMVP_INIT_CKPT="$2" SNMVP_EPISODES="$4" \
    $UV run scripts/train.py pi0_libero_shared --exp-name="$1" \
    --num-train-steps=$STEPS --save-interval=$STEPS --overwrite > "$RD/$1.log" 2>&1
  unset SNMVP_PIN_U
}

# 0) scratch source (no pin) if missing
if [ ! -d "$CKD/snmvp_src_scratch_shared/4999" ]; then
  echo "TRAIN scratch source $(date -u +%H:%M:%S)" >> "$ST"
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled SNMVP_EPISODES="$RD/source_episodes.json" \
    $UV run scripts/train.py pi0_libero_shared --exp-name=snmvp_src_scratch_shared --num-train-steps=5000 --save-interval=2500 --overwrite > "$RD/src_scratch_shared.log" 2>&1
  echo "SCRATCH_SRC_DONE=$? $(date -u +%H:%M:%S)" >> "$ST"
fi

serve_eval() {  # exp prior|"" suite local out
  local exp=$1 prior=$2 suite=$3 local_id=$4 out=$5
  local PORT=$((8050 + RANDOM % 200)) SLOG=$(mktemp /tmp/htoh_serve.XXXX.log)
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 WANDB_MODE=disabled \
    $UV run serve_pca_pin.py --dir "$CKD/$exp/$STEPS" --U "$U" ${prior:+--prior $prior} \
    --config pi0_libero_shared --port $PORT > "$SLOG" 2>&1 &
  local SPID=$!
  local ok=0
  for i in $(seq 1 100); do grep -qiE "listening|Serving .* on port" "$SLOG" && { ok=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 5; done
  if [ $ok -ne 1 ]; then echo "SERVE_FAIL $exp" >> "$ST"; tail -6 "$SLOG" >> "$ST"; kill $SPID 2>/dev/null; return; fi
  sleep 3
  CUDA_VISIBLE_DEVICES=$GPU SNMVP_TASK_ID=$local_id "$LVENV" "$HOME/code/source-noise-mvp/scripts/libero_eval_client.py" \
    --args.host localhost --args.port $PORT --args.task-suite-name "$suite" \
    --args.num-trials-per-task "$TRIALS" --args.no-save-videos --args.out-json "$out" > "${out}.clog" 2>&1
  kill $SPID 2>/dev/null; sleep 2
  [ -f "$out" ] && echo "  succ=$(python3 -c "import json;print(json.load(open('$out')).get('total_success_rate'))" 2>/dev/null)" || { echo "  CLIENT_FAIL"; tail -5 "${out}.clog" >> "$ST"; }
}

# tasks: goal 11 (held-out, language-driven), object 21 (held-out, state-driven)
declare -A SUITE=( [11]=libero_goal [21]=libero_object )
for g in 11 21; do
  su=${SUITE[$g]}
  FS=$RD/fs_hhv_t${g}.json
  SNMVP_FS_TASK=$g SNMVP_FS_OUT=$FS "$HOME/code/openpi/.venv/bin/python" "$RD/make_fs_episodes.py" >> "$ST" 2>&1
  RES=$(SNMVP_G=$g SNMVP_SU=$su "$LVENV" "$RD/resolve_task.py" 2>/dev/null | grep '^RESOLVED' | tail -1)
  LOCAL=$(echo "$RES" | awk '{print $2}')
  echo "TASK g=$g suite=$su local=$LOCAL $(date -u +%H:%M:%S)" >> "$ST"
  [ "${LOCAL:--1}" = "-1" ] && { echo "RESOLVE_FAIL g=$g" >> "$ST"; continue; }
  # adapt both arms
  train "hhv_pin_t${g}"     "$CKD/snmvp_src_pin_vlm/4999"       "$U" "$FS"; echo "ADAPT pin g=$g rc=$? $(date -u +%H:%M:%S)" >> "$ST"
  train "hhv_scratch_t${g}" "$CKD/snmvp_src_scratch_shared/4999" ""   "$FS"; echo "ADAPT scratch g=$g rc=$? $(date -u +%H:%M:%S)" >> "$ST"
  # pin prior on the same demos
  SNMVP_PIN_U=$U SNMVP_EPISODES=$FS SNMVP_PRIOR_OUT=$RD/prior_hhv_t${g}.npz SNMVP_NB=40 \
    CUDA_VISIBLE_DEVICES=$GPU "$HOME/code/openpi/.venv/bin/python" "$RD/fit_prior_hh.py" >> "$ST" 2>&1
  # closed-loop
  echo "EVAL pin g=$g" >> "$ST";     serve_eval "hhv_pin_t${g}"     "$RD/prior_hhv_t${g}.npz" "$su" "$LOCAL" "$RD/hhv_pin_t${g}.json"
  echo "EVAL scratch g=$g" >> "$ST"; serve_eval "hhv_scratch_t${g}" ""                        "$su" "$LOCAL" "$RD/hhv_scratch_t${g}.json"
done
echo "HTOH_ALL_DONE $(date -u +%H:%M:%S)" >> "$ST"
