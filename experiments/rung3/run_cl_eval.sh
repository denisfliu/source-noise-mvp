#!/usr/bin/env bash
# Closed-loop eval of one adapted checkpoint on one LIBERO_10 task.
# args: CKPT  PRIOR|NONE  TASKID  PORT  GPU  OUT  [SUITE]  [TRIALS]
set -uo pipefail
CKPT="$1"; PRIOR="$2"; TASKID="$3"; PORT="$4"; GPU="$5"; OUT="$6"; SUITE="${7:-libero_10}"; TRIALS="${8:-10}"
RD=$HOME/code/source-noise-mvp/experiments/rung3
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
export PYTHONPATH="$HOME/code/openpi/third_party/libero"
export LIBERO_CONFIG_PATH="$HOME/code/libero-config"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU
cd "$HOME/code/openpi"
SLOG=$(mktemp /tmp/serve.XXXX.log)
PRIORARG=""; [ "$PRIOR" != "NONE" ] && PRIORARG="--prior $PRIOR"

CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 WANDB_MODE=disabled \
  uv run serve_pca_pin.py --dir "$CKPT" --U "$RD/pin_U_pca_k5.npy" $PRIORARG --port "$PORT" > "$SLOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
for i in $(seq 1 90); do
  grep -qiE "listening|Serving .* on port" "$SLOG" && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER_DIED $OUT"; tail -8 "$SLOG"; exit 1; }
  sleep 5
done
sleep 3

CUDA_VISIBLE_DEVICES=$GPU SNMVP_TASK_ID=$TASKID \
  examples/libero/.venv/bin/python "$HOME/code/source-noise-mvp/scripts/libero_eval_client.py" \
  --args.host localhost --args.port "$PORT" --args.task-suite-name "$SUITE" \
  --args.num-trials-per-task "$TRIALS" --args.no-save-videos --args.out-json "$OUT" > "${OUT}.clog" 2>&1
RC=$?
kill $SPID 2>/dev/null
if [ -f "$OUT" ]; then
  python3 -c "import json;d=json.load(open('$OUT'));print('CL_RESULT', '$OUT', d.get('total_success_rate'))"
else
  echo "CL_FAIL $OUT rc=$RC"; tail -10 "${OUT}.clog"
fi
