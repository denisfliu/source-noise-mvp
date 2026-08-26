#!/usr/bin/env bash
# Evaluate one PyTorch checkpoint on LIBERO: serve on GPU 1, run the sim
# client, parse success rates, write a JSON record.
#
# Usage: eval_checkpoint.sh <checkpoint_dir> <trials_per_task> <out_json> [task_suite]
# Example:
#   eval_checkpoint.sh ~/code/openpi/checkpoints/pi0_libero/armA_baseline_s42/5000 10 \
#     ~/code/source-noise-mvp/experiments/phase1/results/evals/armA_s42_step5000.json
set -uo pipefail

CKPT="$1"; TRIALS="$2"; OUT="$3"; SUITE="${4:-libero_spatial}"
PORT="${SNMVP_EVAL_PORT:-8000}"

export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
export HF_HOME="$HOME/code/hf-cache" HF_LEROBOT_HOME="$HOME/code/hf-cache/lerobot"
export PYTHONPATH="$HOME/code/openpi/third_party/libero"
export LIBERO_CONFIG_PATH="$HOME/code/libero-config"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1

cd "$HOME/code/openpi"
SLOG=$(mktemp /tmp/snmvp_serve.XXXX.log); CLOG=$(mktemp /tmp/snmvp_client.XXXX.log)

CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint \
  --policy.config pi0_libero --policy.dir "$CKPT" > "$SLOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for i in $(seq 1 60); do
  grep -q "server listening" "$SLOG" && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "EVAL_FINAL=server_died"; tail -5 "$SLOG"; exit 1; }
  sleep 5
done
grep -q "server listening" "$SLOG" || { echo "EVAL_FINAL=server_timeout"; exit 1; }

mkdir -p "$(dirname "$OUT")"
# the sim client renders via EGL on GPU 1; robosuite asserts MUJOCO_EGL_DEVICE_ID
# is listed in CUDA_VISIBLE_DEVICES, so pin both regardless of caller's env
CUDA_VISIBLE_DEVICES=1 \
examples/libero/.venv/bin/python "$HOME/code/source-noise-mvp/scripts/libero_eval_client.py" \
  --args.host localhost --args.port "$PORT" \
  --args.task-suite-name "$SUITE" --args.num-trials-per-task "$TRIALS" \
  --args.no-save-videos --args.out-json "$OUT" > "$CLOG" 2>&1
RC=$?
kill $SERVER_PID 2>/dev/null

[ -f "$OUT" ] || { echo "EVAL_FINAL=no_result rc=$RC"; tail -10 "$CLOG"; exit 1; }
TOTAL=$(python3 -c "import json;print(json.load(open('$OUT'))['total_success_rate'])")
echo "EVAL_FINAL=ok total_success_rate=$TOTAL"
