#!/bin/bash
# One-command policy server for the hardware campaign (GPU box side).
#
#   bash scripts/hw_serve.sh <arm> [--sketch <name>] [--port 8900] [--bind 0.0.0.0|127.0.0.1] [--tag <run tag>]
#
# arms (rows of docs/real_experiments.tsv):
#   baseline   pi0 baseline            gate_scratch3        serve_gate_plain.py
#   ours       source-noise pin        gate_pin_joint_xswap serve_gate_pin_joint.py + pin env + sigma_map_xswap.json
#   noswap     w/o sim-real swap       gate_pin_joint_gmsig3                          + sigma_map_gmsig3.json
# sketch (ours/noswap only): cmpl_denis | cmpl_min4 | cmpl_min4s | tempo06 | tempo10 | tempo15 | orbit | fig8
#   -> SNMVP_PIN_PROMPT=experiments/rung3/sketch_<name>.json (the server carries the sketch; the drone
#      client just flies the matching --task, e.g. compound_left for cmpl_*, right for tempo/orbit/fig8).
#
# The server binds --bind (default 0.0.0.0 so the drone workstation can connect directly over the lab
# network; use 127.0.0.1 + an ssh tunnel if you prefer). Per-replan command log: ~/gate_flights/clog_<tag>.npy.
# Runs in the foreground; Ctrl-C stops it. One server per terminal; change --port to run two.
set -euo pipefail
ARM=${1:?usage: hw_serve.sh <baseline|ours|noswap> [--sketch name] [--port N] [--bind addr] [--tag t]}; shift
SKETCH=""; PORT=8900; BIND=0.0.0.0; TAG=""
while [ $# -gt 0 ]; do case $1 in
  --sketch) SKETCH=$2; shift 2;; --port) PORT=$2; shift 2;; --bind) BIND=$2; shift 2;; --tag) TAG=$2; shift 2;;
  *) echo "unknown arg $1"; exit 2;; esac; done
RD=$(cd "$(dirname "$0")/../experiments/rung3" && pwd)
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
CKROOT=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3
LOGDIR=$HOME/gate_flights; mkdir -p "$LOGDIR"
TAG=${TAG:-${ARM}${SKETCH:+_$SKETCH}_$(date +%Y%m%d_%H%M%S)}
U=$RD/pin_U_mh16.npy
EV=(env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0)
PINENV=(SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1
        SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 CLOG=$LOGDIR/clog_$TAG.npy)
case $ARM in
  baseline) CK=$CKROOT/gate_scratch3/4999; SIG="";;
  ours)     CK=$CKROOT/gate_pin_joint_xswap/4999;  SIG=$RD/sigma_map_xswap.json;;
  noswap)   CK=$CKROOT/gate_pin_joint_gmsig3/4999; SIG=$RD/sigma_map_gmsig3.json;;
  *) echo "arm must be baseline | ours | noswap"; exit 2;;
esac
[ -d "$CK" ] || { echo "checkpoint missing: $CK"; exit 1; }
if [ -n "$SKETCH" ]; then
  [ "$ARM" = baseline ] && { echo "sketches need a pin arm (ours/noswap)"; exit 2; }
  SK=$RD/sketch_$SKETCH.json; [ -f "$SK" ] || { echo "sketch missing: $SK"; exit 1; }
  PINENV+=(SNMVP_PIN_PROMPT=$SK)
fi
echo "== hw_serve: arm=$ARM ckpt=$CK sketch=${SKETCH:-none} bind=$BIND:$PORT tag=$TAG"
echo "== client (drone workstation, dronevla2.0 repo root, branch gate-pin):"
echo "     python run_policy.py gate --task <left|right|center_from_left|center_from_right|compound_left|compound_right> \\"
echo "         --policy_host manaan --policy_port $PORT --trial ${TAG}_t1"
echo "== command log: $LOGDIR/clog_$TAG.npy"
cd "$RD"
if [ "$ARM" = baseline ]; then
  exec "${EV[@]}" "$VENVPY" serve_gate_plain.py --ckpt "$CK" --config pi0_gate --norm "$HFB/assets/gate_nav" --host "$BIND" --port "$PORT"
else
  exec "${EV[@]}" "${PINENV[@]}" SNMVP_SIGMA_MAP="$SIG" "$VENVPY" serve_gate_pin_joint.py --ckpt "$CK" --config pi0_gate \
       --norm "$HFB/assets/gate_nav" --pin-u "$U" --host "$BIND" --port "$PORT"
fi
