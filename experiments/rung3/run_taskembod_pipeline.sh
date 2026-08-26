#!/bin/bash
# Controlled task-by-embodiment grid: collect 3 tasks x 3 arms, then decompose
# transfer into an embodiment axis and a task axis. Launch detached with:
#   setsid bash run_taskembod_pipeline.sh < /dev/null > taskembod_pipeline.log 2>&1 & disown
set -u
cd ~/code/source-noise-mvp/experiments/rung3
UVR="$HOME/.local/bin/uv run --with robosuite==1.4.1 --with mujoco==2.3.7 --with numpy --python 3.11 python"
VENV=../../.venv/bin/python
echo "[te] start $(date -u +%H:%M:%S)"

collect() {  # task arm gpu
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$3 CUDA_VISIBLE_DEVICES=$3 \
    SNMVP_TASK=$1 SNMVP_ARM=$2 $UVR collect_task.py > te_${1}_${2}.log 2>&1
}

jobs=()
for t in bank vertical slalom; do for a in Panda IIWA UR5e; do jobs+=("$t $a"); done; done
n=${#jobs[@]}
for ((i=0; i<n; i+=2)); do
  set -- ${jobs[i]};   [ -s data_taskembod/${1}_${2}.npz ] || collect $1 $2 0 &
  if [ $((i+1)) -lt $n ]; then set -- ${jobs[i+1]}; [ -s data_taskembod/${1}_${2}.npz ] || collect $1 $2 1 & fi
  wait
done
echo "[te] collection done $(date -u +%H:%M:%S)"
grep -h "demo success" te_*.log 2>/dev/null

$VENV taskembod_study.py > taskembod_study.log 2>&1
echo "ALL_DONE $(date -u +%H:%M:%S)" > taskembod.status
echo "[te] finished $(date -u +%H:%M:%S)"
