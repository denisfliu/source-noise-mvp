#!/bin/bash
# Primitive-compliance probe across served arms (2026-09-18): for each arm, start the server on a
# spare port with the agent mode, run tools/agent_primitive_probe.py from dronevla2.0, stop it.
#   bash scripts/agent_probe_arms.sh noswap ours realonly   -> ~/gate_flights/agent_probe/<arm>_sigma*.json
set -uo pipefail
PORT=${PORT:-9250}; EP=${EP:-$HOME/code/source-noise-mvp/experiments/rung3/data_gate_real/ep_0000.npz}
SIGMAS=${SIGMAS:-"0.0"}
cd "$(dirname "$0")/.."
for ARM in "$@"; do
  LOG=$HOME/gate_flights/server_probe_${ARM}_$PORT.log
  setsid bash scripts/hw_serve.sh "$ARM" --port $PORT --tag probe_$ARM </dev/null >"$LOG" 2>&1 &
  timeout 600 grep -m1 -q "ready on" <(tail -n+1 -f "$LOG") || { echo "server for $ARM did not come up"; continue; }
  for S in $SIGMAS; do
    (cd ~/code/dronevla2.0 && /home/dfliu/code/openpi/.venv/bin/python tools/agent_primitive_probe.py --port $PORT --episode "$EP" --tag "$ARM" --sigma "$S" 2>&1 | grep -v "Warning\|\[gate\]")
  done
  PID=$(ss -ltnp | grep ":$PORT " | grep -o "pid=[0-9]*" | cut -d= -f2); [ -n "$PID" ] && kill "$PID"
  timeout 60 bash -c "while ss -ltn | grep -q ':$PORT '; do :; done"
done
echo PROBES_DONE
