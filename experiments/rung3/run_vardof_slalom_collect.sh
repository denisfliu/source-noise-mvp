#!/bin/bash
# Variable-DOF SLALOM (bottlenecked) collection: hard S-weave under OSC_POSE (6-ch) on
# Panda/IIWA/UR5e (set-A) + OSC_POSITION (3-ch) on UR5e (held-out). GPU 0 ONLY, SEQUENTIAL.
# Launch: setsid bash run_vardof_slalom_collect.sh < /dev/null > vardof_slalom_collect.log 2>&1 & disown
set -u
cd ~/code/source-noise-mvp/experiments/rung3
UVR="$HOME/.local/bin/uv run --with robosuite==1.4.1 --with mujoco==2.3.7 --with numpy --python 3.11 python"
echo "[vds] start $(date -u +%H:%M:%S)"
run() {  # ctrl arm
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    SNMVP_CTRL=$1 SNMVP_ARM=$2 $UVR collect_vardof_slalom.py > vds_$1_$2.log 2>&1
}
[ -s data_vardof_slalom/pose_Panda.npz ] || run pose Panda
[ -s data_vardof_slalom/pose_IIWA.npz ]  || run pose IIWA
[ -s data_vardof_slalom/pose_UR5e.npz ]  || run pose UR5e
[ -s data_vardof_slalom/pos_UR5e.npz ]   || run pos  UR5e
echo "[vds] done $(date -u +%H:%M:%S)"
grep -h "demo success" vds_*.log 2>/dev/null
echo "VARDOF_SLALOM_COLLECT_DONE=ok" > vardof_slalom_collect.status
echo "[vds] finished $(date -u +%H:%M:%S)"
