#!/usr/bin/env bash
# Corrected eval-only pass for the head-to-head (the adapted checkpoints are saved at step /1999,
# not /2000). Serves each hh_{pin,scratch}_t{11,21}/1999 with serve_pca_pin (RRR U; pin arm uses its
# state prior, scratch arm unpinned) and runs the LIBERO client. Skips any checkpoint not present.
#   bash run_hh_eval.sh <GPU> <TRIALS>
set -uo pipefail
GPU=${1:-0}; TRIALS=${2:-15}
RD=$HOME/code/source-noise-mvp/experiments/rung3
CKD=$HOME/code/openpi/checkpoints/pi0_libero_shared
U=$RD/pin_U_rrr_k5_shared.npy
UV=$HOME/.local/bin/uv
LVENV=$HOME/code/openpi/examples/libero/.venv/bin/python
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
export PYTHONPATH="$HOME/code/openpi/third_party/libero"
export LIBERO_CONFIG_PATH="$HOME/code/libero-config"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU
cd "$HOME/code/openpi"
ST=$RD/hh_eval.status; echo "HHEVAL_START gpu=$GPU $(date -u +%H:%M:%S)" >> "$ST"
declare -A SUITE=( [11]=libero_goal [21]=libero_object )

serve_eval() {  # ckpt prior|"" suite local out
  local ckpt=$1 prior=$2 suite=$3 local_id=$4 out=$5
  [ -d "$ckpt" ] || { echo "MISSING $ckpt" >> "$ST"; return; }
  local PORT=$((8100 + RANDOM % 300)) SLOG=$(mktemp /tmp/hhev.XXXX.log)
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 WANDB_MODE=disabled \
    $UV run serve_pca_pin.py --dir "$ckpt" --U "$U" ${prior:+--prior $prior} --config pi0_libero_shared --port $PORT > "$SLOG" 2>&1 &
  local SPID=$! ok=0
  for i in $(seq 1 100); do grep -qiE "listening|Serving .* on port" "$SLOG" && { ok=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 5; done
  [ $ok -ne 1 ] && { echo "SERVE_FAIL $ckpt" >> "$ST"; tail -5 "$SLOG" >> "$ST"; kill $SPID 2>/dev/null; return; }
  sleep 3
  CUDA_VISIBLE_DEVICES=$GPU SNMVP_TASK_ID=$local_id "$LVENV" "$HOME/code/source-noise-mvp/scripts/libero_eval_client.py" \
    --args.host localhost --args.port $PORT --args.task-suite-name "$suite" --args.num-trials-per-task "$TRIALS" \
    --args.no-save-videos --args.out-json "$out" > "${out}.clog" 2>&1
  kill $SPID 2>/dev/null; sleep 2
  [ -f "$out" ] && echo "RESULT $(basename $out) succ=$(python3 -c "import json;print(json.load(open('$out')).get('total_success_rate'))" 2>/dev/null)" >> "$ST" || { echo "CLIENT_FAIL $ckpt" >> "$ST"; tail -5 "${out}.clog" >> "$ST"; }
}

for g in 11 21; do
  su=${SUITE[$g]}
  RES=$(SNMVP_G=$g SNMVP_SU=$su "$LVENV" "$RD/resolve_task.py" 2>/dev/null | grep '^RESOLVED' | tail -1)
  LOCAL=$(echo "$RES" | awk '{print $2}'); [ "${LOCAL:--1}" = "-1" ] && { echo "RESOLVE_FAIL g=$g" >> "$ST"; continue; }
  serve_eval "$CKD/hh_pin_t${g}/1999"     "$RD/prior_hh_t${g}.npz" "$su" "$LOCAL" "$RD/hh_pin_t${g}.json"
  serve_eval "$CKD/hh_scratch_t${g}/1999" ""                        "$su" "$LOCAL" "$RD/hh_scratch_t${g}.json"
done
echo "HHEVAL_DONE $(date -u +%H:%M:%S)" >> "$ST"
