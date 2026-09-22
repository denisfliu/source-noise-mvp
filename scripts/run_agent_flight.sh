#!/bin/bash
# One agent-reviewed flight in the simulator (2026-09-20). The policy proposes a command at every replan;
# the reviewer (Claude Code, via experiments/rung3/agent_sim_cli.py) approves it or replaces it, and the
# reasons are burned into the video.  bash scripts/run_agent_flight.sh <tag> <scene> <side> <prompt> [NCH]
set -u
TAG=${1:?tag}; SCENE=${2:?scene}; SIDE=${3:?side}; PROMPT=${4:?prompt}; NCH=${5:-10}; APC=${6:-50}
# ARM=gmsig3|ours|waypoints; START="x,y,z" and STARTYAW=<rad> (rollout yaw convention; the reviewer sees -STARTYAW as its heading) pass through
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; TV=/home/dfliu/code/tv/bin/python; HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy
# ARM (2026-09-22, docs/AGENT_EVAL_PLAN.md): gmsig3 (mixed pin, the flights before the plan) | ours (real-only pin) |
# waypoints (real-only server in decode-only mode: the move's command track is flown as the chunk, no denoising).
ARM=${ARM:-gmsig3}
case $ARM in
  gmsig3)    CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_gmsig3/4999;   SIG=$RD/sigma_map_gmsig3.json;   DEC=0;;
  ours)      CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_realonly/4999; SIG=$RD/sigma_map_realonly.json; DEC=0;;
  waypoints) CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_realonly/4999; SIG=$RD/sigma_map_realonly.json; DEC=1;;
  *) echo "ARM must be gmsig3 | ours | waypoints"; exit 2;;
esac
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$SIG SNMVP_CLOG_FULL=1 SNMVP_PIN_DECODE_ONLY=$DEC"
PORT=${PORT:-9160}; AD=$RUN/agent_sim_$TAG   # one mailbox per flight: two reviewers must never share one
rm -rf $AD; mkdir -p $AD/obs $AD/cmds; echo $ARM > $AD/arm
cd $RD
for p in $(ss -ltnp | grep ":$PORT " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill -9 "$p" 2>/dev/null; done; sleep 3
setsid $EV $PINENV CLOG=$RUN/clog_${TAG}.npy $GPU $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK \
  --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
for k in $(seq 1 200); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT; exit 1; }
setsid env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=$NCH APC=$APC TRIALS=1 VIDEO=1 ${START:+START=$START} ${STARTYAW:+STARTYAW=$STARTYAW} \
  VIDFRAME_STRIDE=4 FPS=9 PROMPT="$PROMPT" AGENT_DIR=$AD SNMVP_PIN_U=$U \
  OUT=$RUN/agent_${TAG}.mp4 TRAJ=$RUN/traj_${TAG}.npy $TV $RD/gate_rollout_batch.py \
  > $RUN/roll_${TAG}.log 2>&1 </dev/null & disown
echo "flight $TAG started; review with: python3 experiments/rung3/agent_sim_cli.py wait --dir $AD"
