#!/usr/bin/env bash
# Both-cases CLOSED-LOOP eval of the frozen RRR source-pin flow, driven by (state) vs (state+lang)
# priors vs no-pin, on in-source goal (language-driven) and object (state-driven) LIBERO tasks.
# Self-contained + SSH-drop safe: launch with
#   setsid bash run_cl_lang.sh </dev/null >cl_lang_driver.log 2>&1 & disown
# Every step appends to cl_lang.status. Per (task,mode): boot serve_pin_lang, run the LIBERO client,
# record success, kill the server. Results -> cl_lang_<mode>_t<g>.json.
set -uo pipefail
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_shared/snmvp_src_pin_rrr/4999
U=$RD/pin_U_rrr_k5_shared.npy
GPU=${SNMVP_GPU:-0}; TRIALS=${1:-12}
LVENV=$HOME/code/openpi/examples/libero/.venv/bin/python
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
export PYTHONPATH="$HOME/code/openpi/third_party/libero"
export LIBERO_CONFIG_PATH="$HOME/code/libero-config"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU
cd "$HOME/code/openpi"
rm -f "$RD/cl_lang.status"
echo "START $(date -u +%H:%M:%S) trials=$TRIALS" >> "$RD/cl_lang.status"

# 1) fit priors once
if [ ! -f "$RD/prior_statelang.npz" ]; then
  SNMVP_CKPT=$CK SNMVP_U=$U CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
    "$HOME/code/openpi/.venv/bin/python" "$RD/fit_lang_prior.py" > "$RD/fit_lang_prior.log" 2>&1
  echo "PRIOR_FIT $(grep -o 'PRIOR_FIT_DONE.*' "$RD/fit_lang_prior.log" | tail -1)" >> "$RD/cl_lang.status"
fi
[ -f "$RD/prior_statelang.npz" ] || { echo "PRIOR_FIT_FAILED" >> "$RD/cl_lang.status"; tail -8 "$RD/fit_lang_prior.log" >> "$RD/cl_lang.status"; exit 1; }

declare -A SUITE=( [12]=libero_goal [15]=libero_goal [20]=libero_object [24]=libero_object )
PORT=8030
for g in 12 15 20 24; do
  su=${SUITE[$g]}
  RES=$(SNMVP_G=$g SNMVP_SU=$su "$LVENV" "$RD/resolve_task.py" 2>/dev/null | grep '^RESOLVED' | tail -1)
  LOCAL=$(echo "$RES" | awk '{print $2}'); LANGCSV=$(echo "$RES" | awk '{print $3}')
  echo "TASK g=$g suite=$su local=$LOCAL onehot=$LANGCSV" >> "$RD/cl_lang.status"
  [ "${LOCAL:-'-1'}" = "-1" ] && { echo "RESOLVE_FAIL g=$g" >> "$RD/cl_lang.status"; continue; }
  for mode in none state statelang; do
    PORT=$((PORT+1))
    PRIORARG=""; LANGARG=""
    [ "$mode" = state ]     && PRIORARG="--prior $RD/prior_state.npz"
    [ "$mode" = statelang ] && { PRIORARG="--prior $RD/prior_statelang.npz"; LANGARG="--langfeat $LANGCSV"; }
    SLOG=$(mktemp /tmp/serve_lang.XXXX.log)
    CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 WANDB_MODE=disabled \
      uv run "$RD/serve_pin_lang.py" --dir "$CK" --U "$U" $PRIORARG $LANGARG --port "$PORT" > "$SLOG" 2>&1 &
    SPID=$!
    ok=0
    for i in $(seq 1 100); do
      grep -qiE "listening|Serving .* on port" "$SLOG" && { ok=1; break; }
      kill -0 $SPID 2>/dev/null || break
      sleep 5
    done
    if [ $ok -ne 1 ]; then echo "SERVE_FAIL g=$g mode=$mode" >> "$RD/cl_lang.status"; tail -6 "$SLOG" >> "$RD/cl_lang.status"; kill $SPID 2>/dev/null; sleep 2; continue; fi
    sleep 3
    OUT=$RD/cl_lang_${mode}_t${g}.json
    CUDA_VISIBLE_DEVICES=$GPU SNMVP_TASK_ID=$LOCAL \
      "$LVENV" "$HOME/code/source-noise-mvp/scripts/libero_eval_client.py" \
      --args.host localhost --args.port "$PORT" --args.task-suite-name "$su" \
      --args.num-trials-per-task "$TRIALS" --args.no-save-videos --args.out-json "$OUT" > "${OUT}.clog" 2>&1
    kill $SPID 2>/dev/null; sleep 2
    if [ -f "$OUT" ]; then
      sr=$(python3 -c "import json;print(json.load(open('$OUT')).get('total_success_rate'))" 2>/dev/null)
      echo "RESULT g=$g mode=$mode success=$sr $(date -u +%H:%M:%S)" >> "$RD/cl_lang.status"
    else
      echo "CLIENT_FAIL g=$g mode=$mode" >> "$RD/cl_lang.status"; tail -6 "${OUT}.clog" >> "$RD/cl_lang.status"
    fi
  done
done
echo "CL_LANG_ALL_DONE $(date -u +%H:%M:%S)" >> "$RD/cl_lang.status"
