#!/bin/bash
# One parameterised 10-trial evaluation, for either serving path. Exists because the rollouts turned
# out to be non-reproducible run-to-run (identical config and seed differ by up to 0.63 m over ~400
# closed-loop steps), so every conclusion resting on a 5-trial cell has to be re-measured at 10 before
# it can be compared with anything.
#
#   run_eval10.sh NAME MODE CKPT GPU PORT TRIALS [EXTRA_ENV] [PRIOR]
#
# MODE=joint      command head read from the flow's own checkpoint (serve_gate_pin_joint.py)
# MODE=langprior  external prior file (serve_gate_pin_langprior.py); pass PRIOR
set -u
NAME=$1; MODE=$2; CK=$3; GPU=$4; PORT=$5; TRIALS=$6; EXTRA=${7:-}; PRIOR=${8:-}
RUN=/home/ubuntu/ctxrun
RD=/home/ubuntu/code/source-noise-mvp/experiments/rung3
PY=/home/ubuntu/code/openpi/.venv/bin/python; HFB=/home/ubuntu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_gate_rrr_k5.npy
export PATH=/tmp/tv/bin:/usr/local/cuda-12.8/bin:$PATH; export CUDA_HOME=/usr/local/cuda-12.8
rm -f $RUN/ev_$NAME.done $RUN/ev_${NAME}_scores.txt
[ -d "$CK/params" ] || { echo CKPT_MISSING > $RUN/ev_$NAME.done; exit 1; }
for k in $(seq 1 720); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$u" -lt 20000 ] && break; sleep 30
done
cd $RD

if [ "$MODE" = "joint" ]; then
  SRV=serve_gate_pin_joint.py
  BASE="SNMVP_HEAD=1 SNMVP_PIN_U=$U"
  HS=""; case "$EXTRA" in *SNMVP_HEAD_STATE=1*) HS="--head-state";; esac
  # readout gate first: a broken readout would otherwise look like a bad command source
  $EV $BASE $EXTRA CUDA_VISIBLE_DEVICES=$GPU $PY $RD/joint_head.py --ckpt $CK --pin-u $U \
    --norm $HFB/assets/gate_nav --n 8 $HS --check > $RUN/ev_${NAME}_check.log 2>&1
  python3 - "$RUN/ev_${NAME}_check.log" > $RUN/ev_${NAME}_scores.txt << 'PYEOF'
import sys
rows = [l.split() for l in open(sys.argv[1]) if l.startswith(("left", "right", "center"))]
r2 = [float(r[2]) for r in rows if len(r) >= 3]
print(f"== readout gate: {len(r2)} tasks, min per-task c-R2 {min(r2) if r2 else float('nan'):+.4f}")
print("== GATE PASS" if r2 and min(r2) > 0.5 else "== GATE FAIL")
PYEOF
  grep -qa "GATE PASS" $RUN/ev_${NAME}_scores.txt || { echo GATE_FAILED > $RUN/ev_$NAME.done; exit 1; }
  ARGS="--ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --port $PORT"
else
  SRV=serve_gate_pin_langprior.py
  BASE="LATCH_N=0"
  ARGS="--ckpt $CK --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U --prior $PRIOR --port $PORT"
  : > $RUN/ev_${NAME}_scores.txt
fi

for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
sleep 3
setsid $EV $BASE $EXTRA CLOG=$RUN/clog_ev_$NAME.npy CUDA_VISIBLE_DEVICES=$GPU $PY $RD/$SRV $ARGS \
  >> $RUN/sv_ev_$NAME.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT > $RUN/ev_$NAME.done; exit 1; }
# batch client: scene loads once per side, sides run in parallel; VIDEO=1 (stride-4, fps 9)
for side in left right; do
  env CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT SIDE=$side SCENE=$side NCH=8 APC=50 TRIALS=$TRIALS \
    OUT=$RUN/overlay_ev${NAME}_${side}_{t}.mp4 TRAJ=$RUN/traj_ev${NAME}_${side}_{t}.npy \
    /tmp/tv/bin/python $RD/gate_rollout_batch.py > $RUN/roll_ev${NAME}_${side}.log 2>&1 &
done
wait
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
{ for side in left right; do
    echo "== $NAME, APC=50, $side"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $PY $RD/gate_success.py \
      --traj $RUN/traj_ev${NAME}_${side}_*.npy --side $side
    /tmp/tv/bin/python $RD/gate_clearance.py --scene $side --traj $RUN/traj_ev${NAME}_${side}_*.npy
  done } >> $RUN/ev_${NAME}_scores.txt 2>&1
grep -qa "clearance-clean" $RUN/ev_${NAME}_scores.txt && echo DONE > $RUN/ev_$NAME.done \
  || echo SCORE_FAILED > $RUN/ev_$NAME.done
