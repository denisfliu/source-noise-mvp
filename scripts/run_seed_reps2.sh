#!/bin/bash
# Seed-rep round 2 (Denis, 2026-08-26):
#   Phase A — minimal-sketch replication on the seed-7 checkpoint (gmsig3s7): CMPL min4
#             sigma=0, CMPL min4 sigma=0.5, CMPR min5 sigma=0 — does the 4-5-point result
#             survive a different training seed?
#   Phase B — scratch control seed rep: gate_scratch3s7 (--seed=7, plain pi0, no SNMVP env).
#   Phase C — its six cells via run_sixcell_plain.sh (plain serve).
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
U=$RD/pin_U_mh16.npy
BASE="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U"
HEADENV="SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_gmsig3s7.json"
CKS7=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_gmsig3s7/4999
SRV=serve_gate_pin_joint.py
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
PORT=9080
rm -f $RUN/seed_reps2.done $RUN/seed_reps2_phaseA.done $RUN/sketch_s7_scores.txt
cd $RD

cell () { # tag sketch_json scene client_prompt
  local TAG=$1 SK=$2 SCENE=$3 CPROMPT=$4
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  setsid $EV $BASE $HEADENV SNMVP_PIN_PROMPT=$RD/$SK CLOG=$RUN/clog_${TAG}.npy \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0 \
    $VENVPY $RD/$SRV --ckpt $CKS7 --config pi0_gate --norm $HFB/assets/gate_nav --pin-u $U \
    --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $RUN/sketch_s7_scores.txt; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=${SCENE%%_*} SCENE=$SCENE NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    PROMPT="$CPROMPT" TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_${TAG}.log 2>&1
  for p in $(pgrep -f "$SRV --ckpt .* --port $PORT"); do kill -9 "$p" 2>/dev/null; done
  { echo "== seed-7 checkpoint, sketch rep: $TAG"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py \
      --traj $RUN/traj_${TAG}_*.npy --side $SCENE
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
  } >> $RUN/sketch_s7_scores.txt 2>&1
}
cell s7m4_cmpl sketch_cmpl_min4.json left_and_center "$CMP_L"
cell s7m4s_cmpl sketch_cmpl_min4s.json left_and_center "$CMP_L"
cell s7m5_cmpr sketch_cmpr_min5.json right_and_center "$CMP_R"
echo PHASE_A_DONE > $RUN/seed_reps2_phaseA.done

# ---------- Phase B: scratch seed-7 training (plain pi0, NO SNMVP env) ----------
[ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -ge 10 ] \
  || { echo DISK_GUARD_FAILED > $RUN/seed_reps2.done; exit 1; }
for k in $(seq 1 40); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$u" -lt 2000 ] && break; sleep 15
done
cd /home/dfliu/code/openpi-snmvp
env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0 \
  /home/dfliu/code/openpi/.venv/bin/python scripts/train.py pi0_gate3 \
  --exp-name=gate_scratch3s7 --num-train-steps=5000 --lr-schedule.decay-steps=1000000 \
  --save-interval=5000 --seed=7 --no-wandb-enabled --overwrite \
  > $RUN/arm_scr3s7_train.log 2>&1
rm -rf /home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch3s7/*/train_state
echo PHASE_B_DONE >> $RUN/seed_reps2_phaseA.done

# ---------- Phase C: plain six cells ----------
bash /home/dfliu/code/source-noise-mvp/scripts/run_sixcell_plain.sh scr3s7 \
  /home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch3s7/4999 9082 10
echo DONE > $RUN/seed_reps2.done
