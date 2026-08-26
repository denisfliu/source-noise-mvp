#!/bin/bash
# Compound prompts CORRECTED (Denis, 2026-08-13): the second clause now carries the approach
# direction ("...then go through the center gate from the left/right..."), matching the trained
# CFL/CFR phrasing — the old wording ("then through the center gate") exists in no training task,
# so the second hop was out-of-vocabulary. Judge geometry agrees: gate-2 direction signs equal
# CFL/CFR's. Re-screens b2lam03, c2, mh16 (5 trials/scene, VIDEO=0), sequential on GPU0.
set -u
RUN=/home/ubuntu/ctxrun; GPU=0; PORT=8925
RD=/home/ubuntu/code/source-noise-mvp/experiments/rung3
PY=/home/ubuntu/code/openpi/.venv/bin/python; HFB=/home/ubuntu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false"
export PATH=/tmp/tv/bin:/usr/local/cuda-12.8/bin:$PATH; export CUDA_HOME=/usr/local/cuda-12.8
CMP_L="go through the gate on the left, then go through the center gate from the left and hover over the stuffed animal"
CMP_R="go through the gate on the right, then go through the center gate from the right and hover over the stuffed animal"
UD=$RD/pin_U_gate_rrr_k5.npy; UV=$RD/pin_U_vla_base_k5.npy; UM=$RD/pin_U_mh16.npy
CKB=/home/ubuntu/code/openpi/checkpoints/pi0_gate
rm -f $RUN/cmpfix.done
run_arm () {  # $1=name $2=ckpt $3=U $4=extra
  for k in $(seq 1 200); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    [ "$u" -lt 40000 ] && break; sleep 30
  done
  cd $RD
  for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV SNMVP_HEAD=1 SNMVP_PIN_U=$3 SNMVP_HEAD_DETACH=0 $4 CUDA_VISIBLE_DEVICES=$GPU $PY \
    $RD/serve_gate_pin_joint.py --ckpt $2 --config pi0_gate --norm $HFB/assets/gate_nav \
    --pin-u $3 --port $PORT >> $RUN/sv_cmpfix_$1.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  env CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT SIDE=left SCENE=left_and_center NCH=14 APC=50 TRIALS=5 \
    VIDEO=0 PROMPT="$CMP_L" TRAJ=$RUN/traj_cmpfix_$1_l_{t}.npy \
    /tmp/tv/bin/python $RD/gate_rollout_batch.py > $RUN/roll_cmpfix_$1_l.log 2>&1
  env CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT SIDE=right SCENE=right_and_center NCH=14 APC=50 TRIALS=5 \
    VIDEO=0 PROMPT="$CMP_R" TRAJ=$RUN/traj_cmpfix_$1_r_{t}.npy \
    /tmp/tv/bin/python $RD/gate_rollout_batch.py > $RUN/roll_cmpfix_$1_r.log 2>&1
  for p in $(pgrep -f "serve_gate_pin_joint.py --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
}
run_arm b2lam03 $CKB/gate_pin_joint_b2lam03/4999 $UD "SNMVP_HEAD_LAM=0.3"
run_arm c2 $CKB/gate_pin_joint_c2/4999 $UV "SNMVP_FLOW_DETACH=1"
run_arm mh16 $CKB/gate_pin_joint_mh16/4999 $UM "SNMVP_HEAD_LAM=0.3"
{ for a in b2lam03 c2 mh16; do
    echo "== cmpfix $a left_and_center (directional prompt)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $PY $RD/gate_success.py \
      --traj $RUN/traj_cmpfix_${a}_l_*.npy --side left_and_center
    echo "== cmpfix $a right_and_center (directional prompt)"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $PY $RD/gate_success.py \
      --traj $RUN/traj_cmpfix_${a}_r_*.npy --side right_and_center
  done } > $RUN/cmpfix_scores.txt 2>&1
echo DONE > $RUN/cmpfix.done
