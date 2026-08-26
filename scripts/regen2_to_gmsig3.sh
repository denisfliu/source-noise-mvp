#!/bin/bash
# Orchestrator (Denis "do both", 2026-08-23): after gmsig2's chain releases the GPU and regen2
# phase A finishes, render the kept 200 -> build local/gate_nav3 -> extract npz mirrors ->
# train gmsig3 (identical trust-dial recipe on pi0_gate3). Its post chain runs separately
# (run_gmsig3_post.sh, gated on the checkpoint). Disk guard before the render.
set -u
FAL=/home/dfliu/code/falsify
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
RUN=/home/dfliu/ctxrun
OUT=$FAL/runs/regen2
for k in $(seq 1 1440); do
  [ -f $RUN/arm_gmsig2.done ] && grep -qa PHASE_A_DONE $RUN/regen2_phaseA.log 2>/dev/null && break
  sleep 60
done
[ -f $RUN/arm_gmsig2.done ] || { echo GMSIG2_WAIT_TIMEOUT > $RUN/regen2_pipeline.done; exit 1; }
[ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -ge 45 ] \
  || { echo DISK_LOW_ABORT > $RUN/regen2_pipeline.done; exit 1; }
for k in $(seq 1 120); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$u" -lt 2000 ] && break; sleep 60
done

# phase B: render kept 200
cd $FAL && source tools/env.sh
export PATH=$FAL/.venv/bin:$PATH
for spec in through_left_gate:left_gate through_right_gate:right_gate \
            through_center_gate_from_left:center_gate through_center_gate_from_right:center_gate; do
  CO=${spec%%:*}; SC=${spec##*:}
  mkdir -p $OUT/staging/$CO
  PYTHONPATH=src:external/FiGS/src:external/splatnav .venv/bin/python -m falsify.cli.export_training_data \
    --trajectories-dir $OUT/kept/$CO --scene configs/scenes/$SC.yaml \
    --frame configs/frames/carl_dual.yaml --embodiment configs/embodiments/carl_dual_mocap.yaml \
    --out $OUT/staging/$CO >> $RUN/regen2_render_$CO.log 2>&1
  N=$(find $OUT/staging/$CO -name "*.parquet" | wc -l)
  echo "[render2] $CO: $N parquets" >> $RUN/regen2_pipeline.log
  [ "$N" -eq 50 ] || { echo "RENDER_SHORT_$CO ($N/50)" > $RUN/regen2_pipeline.done; exit 1; }
done

# gate_nav3 build + mirrors
cd $RD
env PYTHONPATH=/home/dfliu/code/openpi-snmvp/src /home/dfliu/code/openpi/.venv/bin/python \
  build_gate_nav3.py >> $RUN/gate_nav3_build.log 2>&1
grep -qa GATE_NAV3_DONE $RUN/gate_nav3_build.log || { echo BUILD3_FAILED > $RUN/regen2_pipeline.done; exit 1; }
env EPS=$RD/gate_synth_eps3.json OUT=$RD/data_gate_synth3 REPO=local/gate_nav3 \
  PYTHONPATH=/home/dfliu/code/openpi-snmvp/src /home/dfliu/code/openpi/.venv/bin/python \
  gate_extract_raw.py >> $RUN/extract_synth3.log 2>&1 &
EXTRACT_PID=$!

# gmsig3 training (mirrors finish long before the readout gate needs them)
cd /home/dfliu/code/openpi-snmvp
env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src SNMVP_HEAD=1 \
  SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$RD/pin_U_mh16.npy SNMVP_HEAD_DETACH=0 \
  SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 \
  SNMVP_PIN_NOISE_COND=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0 \
  /home/dfliu/code/openpi/.venv/bin/python scripts/train.py pi0_gate3 \
  --exp-name=gate_pin_joint_gmsig3 --num-train-steps=5000 --lr-schedule.decay-steps=1000000 \
  --save-interval=5000 --seed=42 --no-wandb-enabled --overwrite \
  > $RUN/arm_gmsig3_train.log 2>&1
wait $EXTRACT_PID 2>/dev/null
echo PIPELINE_TRAIN_DONE > $RUN/regen2_pipeline.done
