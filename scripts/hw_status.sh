#!/bin/bash
# Status / cleanup / self-test for gate policy servers on manaan.
#
#   bash scripts/hw_status.sh                 # running servers (pid, port, checkpoint, sketch), GPU memory, ports
#   bash scripts/hw_status.sh --kill 8900     # stop whatever serves on that port (by pid; never by name pattern)
#   bash scripts/hw_status.sh --test 8900 [task] [episode.npz]
#                                             # dry client against a local server: prints replans + latency
set -uo pipefail
RD=$(cd "$(dirname "$0")/../experiments/rung3" && pwd)
DRONE=${DRONEVLA_DIR:-$HOME/code/dronevla2.0}
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
case ${1:-} in
  --kill)
    PORT=${2:?port}; PIDS=$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$" {print $6}' | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
    [ -z "$PIDS" ] && { echo "nothing listening on $PORT"; exit 0; }
    for p in $PIDS; do echo "killing pid $p ($(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-90))"; kill "$p"; done
    sleep 3; ss -ltn | grep -q ":$PORT " && { echo "still up, forcing"; for p in $PIDS; do kill -9 "$p" 2>/dev/null; done; }
    echo "port $PORT free"; exit 0;;
  --test)
    PORT=${2:-8900}; TASK=${3:-left}; EP=${4:-$RD/data_gate_real/ep_0000.npz}
    [ -f "$DRONE/tools/gate_dry_client.py" ] || { echo "dronevla2.0 (branch gate-pin) not found at $DRONE; set DRONEVLA_DIR"; exit 1; }
    cd "$DRONE" && exec "$VENVPY" tools/gate_dry_client.py --host 127.0.0.1 --port "$PORT" --task "$TASK" --episode "$EP" --replans 6;;
esac
echo "== gate policy servers on $(hostname) ($(hostname -I | awk '{print $1}'))"
FOUND=0
for p in $(pgrep -f 'serve_gate_(pin_joint|plain|sdedit|pin)\.py' 2>/dev/null); do
  FOUND=1; CMD=$(tr '\0' ' ' < /proc/$p/cmdline); ENVV=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null)
  PORT=$(echo "$CMD" | grep -o -- '--port [0-9]*' | awk '{print $2}'); CK=$(echo "$CMD" | grep -o -- '--ckpt [^ ]*' | awk '{print $2}')
  SK=$(echo "$ENVV" | grep '^SNMVP_PIN_PROMPT=' | cut -d= -f2-); SM=$(echo "$ENVV" | grep '^SNMVP_SIGMA_MAP=' | cut -d= -f2-)
  ADV=$(echo "$ENVV" | grep -E '^SNMVP_(PIN_ADVICE|PIN_REASON|PIN_OFF|PIN_DECODE_ONLY|INTENT_WS)=' | tr '\n' ' ')
  START=$(ps -o lstart= -p "$p"); UP=$(ss -ltn | grep -q ":$PORT " && echo listening || echo "NOT listening (loading?)")
  echo "pid $p  port ${PORT:-?}  $UP  since $START"
  echo "    script: $(echo "$CMD" | grep -o 'serve_gate_[a-z_]*\.py')   ckpt: ${CK##*/pi0_gate3/}"
  echo "    sigma map: ${SM:-none}   sketch: ${SK:-none}   ${ADV:+other modes: $ADV}"
done
[ $FOUND = 0 ] && echo "no gate servers running"
echo "== GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo "== other GPU processes:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/    /'
echo "== recent command logs: $(ls -t "$HOME"/gate_flights/clog_*.npy 2>/dev/null | head -3 | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
