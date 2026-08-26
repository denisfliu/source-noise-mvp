#!/bin/bash
# Local-4090 port of run_eval10.sh: one parameterised 10-trial left/right evaluation for either
# serving path. Exists because rollouts are non-reproducible run-to-run (up to 0.63 m over ~400
# steps), so every conclusion resting on a 5-trial cell must be re-measured at 10 before comparison.
#
#   run_eval10_local.sh NAME MODE CKPT PORT TRIALS [EXTRA_ENV] [PRIOR]
#
# MODE=joint      command head read from the flow's own checkpoint (serve_gate_pin_joint.py);
#                 EXTRA must name the head variant the checkpoint was trained with
#                 (e.g. SNMVP_HEAD_GMM=1) or module construction won't match at restore.
# MODE=langprior  external prior file (serve_gate_pin_langprior.py); pass PRIOR
#
# Differences from the box script, all deliberate:
#  - single GPU: waits for the card to be genuinely idle (<2 GB used, up to 12 h — an arm chain
#    ahead of us can hold it that long) so it never colocates with a training job;
#  - the JAX server takes 45% of the card (preallocated) so the two torch render clients fit
#    beside it on 24 GB;
#  - the two clients keep the 120 s stagger (the box's eval10 originally lacked it and lost
#    gen1lam1's right side to the simultaneous-handshake race, 2026-08-13);
#  - disk guards: refuses to start under 5 GB free, and clears this NAME's previous
#    traj/overlay/clog outputs first — scoring globs traj_ev${NAME}_*, so stale trials from a
#    longer earlier run would otherwise contaminate the verdict.
set -u
NAME=$1; MODE=$2; CK=$3; PORT=$4; TRIALS=$5; EXTRA=${6:-}; PRIOR=${7:-}
GPU=0
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_gate_rrr_k5.npy
mkdir -p $RUN
rm -f $RUN/ev_$NAME.done $RUN/ev_${NAME}_scores.txt
[ -d "$CK/params" ] || { echo CKPT_MISSING > $RUN/ev_$NAME.done; exit 1; }
[ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -ge 5 ] \
  || { echo DISK_FULL > $RUN/ev_$NAME.done; exit 1; }
rm -f $RUN/traj_ev${NAME}_*.npy $RUN/overlay_ev${NAME}_*.mp4 $RUN/clog_ev_$NAME.npy

for k in $(seq 1 720); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$u" -lt 2000 ] && break; sleep 60
done
u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
[ "$u" -lt 2000 ] || { echo GPU_BUSY_TIMEOUT > $RUN/ev_$NAME.done; exit 1; }
cd $RD

if [ "$MODE" = "joint" ]; then
  SRV=serve_gate_pin_joint.py
  BASE="SNMVP_HEAD=1 SNMVP_PIN_U=$U"
  HS=""; case "$EXTRA" in *SNMVP_HEAD_STATE=1*) HS="--head-state";; esac
  # readout gate first: a broken readout would otherwise look like a bad command source
  $EV $BASE $EXTRA CUDA_VISIBLE_DEVICES=$GPU $VENVPY $RD/joint_head.py --ckpt $CK --pin-u $U \
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
setsid $EV $BASE $EXTRA CLOG=$RUN/clog_ev_$NAME.npy XLA_PYTHON_CLIENT_PREALLOCATE=true \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=$GPU $VENVPY $RD/$SRV $ARGS \
  >> $RUN/sv_ev_$NAME.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
ss -ltn | grep -q ":$PORT " || { echo SERVER_TIMEOUT > $RUN/ev_$NAME.done; exit 1; }
# batch clients: scene loads once per side; claim tier keeps VIDEO=1 (stride-4 frames, fps 9)
for side in left right; do
  env CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT SIDE=$side SCENE=$side NCH=8 APC=50 TRIALS=$TRIALS \
    OUT=$RUN/overlay_ev${NAME}_${side}_{t}.mp4 TRAJ=$RUN/traj_ev${NAME}_${side}_{t}.npy \
    $TV $RD/gate_rollout_batch.py > $RUN/roll_ev${NAME}_${side}.log 2>&1 &
  sleep 120
done
wait
for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
{ for side in left right; do
    echo "== $NAME, APC=50, $side"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
      --traj $RUN/traj_ev${NAME}_${side}_*.npy --side $side
    $TV $RD/gate_clearance.py --scene $side --traj $RUN/traj_ev${NAME}_${side}_*.npy
  done } >> $RUN/ev_${NAME}_scores.txt 2>&1
grep -qa "clearance-clean" $RUN/ev_${NAME}_scores.txt && echo DONE > $RUN/ev_$NAME.done \
  || echo SCORE_FAILED > $RUN/ev_$NAME.done
