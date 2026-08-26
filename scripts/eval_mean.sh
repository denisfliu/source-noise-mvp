#!/usr/bin/env bash
# Eval a checkpoint under the MEAN (mode-averaging) server on LIBERO.
# Usage: eval_mean.sh <checkpoint_dir> <trials> <out_json> <mean_k> [suite]
set -uo pipefail
CKPT="$1"; TRIALS="$2"; OUT="$3"; K="$4"; SUITE="${5:-libero_spatial}"
PORT="${SNMVP_EVAL_PORT:-8000}"; GPU="${SNMVP_GPU:-1}"; REPLAN="${SNMVP_REPLAN:-5}"
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
export HF_HOME="$HOME/code/hf-cache" HF_LEROBOT_HOME="$HOME/code/hf-cache/lerobot"
export PYTHONPATH="$HOME/code/openpi/third_party/libero"
export LIBERO_CONFIG_PATH="$HOME/code/libero-config"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU
cd "$HOME/code/openpi"
SLOG=$(mktemp /tmp/snmvp_serve.XXXX.log); CLOG=$(mktemp /tmp/snmvp_client.XXXX.log)
echo "serve log $SLOG ; client log $CLOG"
CUDA_VISIBLE_DEVICES=$GPU uv run python "$HOME/code/source-noise-mvp/scripts/serve_mean.py" \
  --config pi0_libero --dir "$CKPT" --port "$PORT" --mean_k "$K" > "$SLOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT
for i in $(seq 1 60); do
  grep -q "server listening" "$SLOG" && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "EVAL_FINAL=server_died"; tail -8 "$SLOG"; exit 1; }
  sleep 5
done
grep -q "server listening" "$SLOG" || { echo "EVAL_FINAL=server_timeout"; tail -8 "$SLOG"; exit 1; }
mkdir -p "$(dirname "$OUT")"
CUDA_VISIBLE_DEVICES=$GPU examples/libero/.venv/bin/python \
  "$HOME/code/source-noise-mvp/scripts/libero_eval_client.py" \
  --args.host localhost --args.port "$PORT" --args.task-suite-name "$SUITE" \
  --args.num-trials-per-task "$TRIALS" --args.replan-steps "$REPLAN" --args.no-save-videos --args.out-json "$OUT" > "$CLOG" 2>&1
RC=$?
kill $SERVER_PID 2>/dev/null
[ -f "$OUT" ] || { echo "EVAL_FINAL=no_result rc=$RC"; tail -12 "$CLOG"; exit 1; }
TOTAL=$(python3 -c "import json;print(json.load(open('$OUT'))['total_success_rate'])")
echo "EVAL_FINAL=ok mean_k=$K total_success_rate=$TOTAL"
