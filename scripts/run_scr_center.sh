#!/bin/bash
# Denis, 2026-08-12: (a) scratch control on CFL/CFR + both compound scenes; (b) b2lam03 (lam=0.3
# coupled) on the center scenes at APC=25 (NCH doubled to 20 so executed steps match the APC=50
# cells). VIDEO=0 throughout (trajectory-only screens; re-fly with video if anything needs review).
# Uses gate_rollout_batch.py — first batch-mode battery. Runs alongside the C2 chain's extraction;
# the C2 training step waits on GPU memory, so this finishes first.
set -u
RUN=/home/ubuntu/ctxrun; GPU=0; PSCR=8916; PJNT=8914
RD=/home/ubuntu/code/source-noise-mvp/experiments/rung3
PY=/home/ubuntu/code/openpi/.venv/bin/python; HFB=/home/ubuntu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_gate_rrr_k5.npy
CKJ=/home/ubuntu/code/openpi/checkpoints/pi0_gate/gate_pin_joint_b2lam03/4999
export PATH=/tmp/tv/bin:/usr/local/cuda-12.8/bin:$PATH; export CUDA_HOME=/usr/local/cuda-12.8
CFL="go through the center gate from the left and hover over the stuffed animal"
CFR="go through the center gate from the right and hover over the stuffed animal"
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
rm -f $RUN/scr_center.done $RUN/scr_center_scores.txt
cd $RD

setsid $EV CUDA_VISIBLE_DEVICES=$GPU $PY $RD/serve_gate_pin_classic.py --mode scratch --port $PSCR \
  >> $RUN/sv_scr_center.log 2>&1 </dev/null & disown
setsid $EV SNMVP_HEAD=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 CUDA_VISIBLE_DEVICES=$GPU $PY \
  $RD/serve_gate_pin_joint.py --ckpt $CKJ --config pi0_gate --norm $HFB/assets/gate_nav \
  --pin-u $U --port $PJNT >> $RUN/sv_l03a25.log 2>&1 </dev/null & disown
for k in $(seq 1 150); do ss -ltn | grep -q ":$PSCR " && ss -ltn | grep -q ":$PJNT " && break; sleep 3; done
ss -ltn | grep -q ":$PSCR " || { echo SCR_SERVER_TIMEOUT > $RUN/scr_center.done; exit 1; }
ss -ltn | grep -q ":$PJNT " || { echo JNT_SERVER_TIMEOUT > $RUN/scr_center.done; exit 1; }

roll () {  # $1=port $2=tag $3=side $4=scene $5=prompt $6=nch $7=apc $8=trials
  env CUDA_VISIBLE_DEVICES=$GPU PORT=$1 SIDE=$3 SCENE=$4 NCH=$6 APC=$7 TRIALS=$8 VIDEO=0 \
    PROMPT="$5" TRAJ=$RUN/traj_$2_{t}.npy \
    /tmp/tv/bin/python $RD/gate_rollout_batch.py > $RUN/roll_$2.log 2>&1
}
# scratch: center pair first (parallel), then compound pair (parallel)
roll $PSCR scr_ctr_cfl left  center "$CFL" 10 50 10 &
roll $PSCR scr_ctr_cfr right center "$CFR" 10 50 10 &
# b2lam03 at APC=25 in parallel on its own server
roll $PJNT l03a25_cfl left  center "$CFL" 20 25 10 &
roll $PJNT l03a25_cfr right center "$CFR" 20 25 10 &
wait
roll $PSCR scr_cmp_left  left  left_and_center  "$CMP_L" 14 50 5 &
roll $PSCR scr_cmp_right right right_and_center "$CMP_R" 14 50 5 &
wait
for port in $PSCR $PJNT; do
  for p in $(pgrep -f "serve_gate_pin.*--port $port"); do kill -9 "$p" 2>/dev/null; done
done

{ for spec in "scr_ctr_cfl center_from_left center" "scr_ctr_cfr center_from_right center" \
              "l03a25_cfl center_from_left center" "l03a25_cfr center_from_right center" \
              "scr_cmp_left left_and_center -" "scr_cmp_right right_and_center -"; do
    set -- $spec; TAG=$1; JUDGE=$2; SCN=$3
    echo "== $TAG (judge: $JUDGE)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $PY $RD/gate_success.py \
      --traj $RUN/traj_${TAG}_*.npy --side $JUDGE
    [ "$SCN" != "-" ] && /tmp/tv/bin/python $RD/gate_clearance.py --scene $SCN \
      --traj $RUN/traj_${TAG}_*.npy
  done } > $RUN/scr_center_scores.txt 2>&1
grep -qa "success" $RUN/scr_center_scores.txt && echo DONE > $RUN/scr_center.done \
  || echo SCORE_FAILED > $RUN/scr_center.done
