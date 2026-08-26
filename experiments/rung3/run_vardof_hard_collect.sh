#!/bin/bash
# Variable-DOF gate collection: hard position detour-reach under OSC_POSE (6-ch) on
# Panda/IIWA/UR5e (set-A) and OSC_POSITION (3-ch) on UR5e (held-out). Held-out = same
# arm as an in-set body, so the transfer tests the pure CONTROLLER/DOF change.
# GPU 0 ONLY, SEQUENTIAL (other GPU reserved).
# Launch: setsid bash run_vardof_hard_collect.sh < /dev/null > vardof_hard_collect.log 2>&1 & disown
set -u
cd ~/code/source-noise-mvp/experiments/rung3
UVR="$HOME/.local/bin/uv run --with robosuite==1.4.1 --with mujoco==2.3.7 --with numpy --python 3.11 python"
echo "[vdh] start $(date -u +%H:%M:%S)"

run() {  # ctrl arm
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    SNMVP_CTRL=$1 SNMVP_ARM=$2 $UVR collect_vardof_hard.py > vdh_$1_$2.log 2>&1
}

[ -s data_vardof_hard/pose_Panda.npz ] || run pose Panda
[ -s data_vardof_hard/pose_IIWA.npz ]  || run pose IIWA
[ -s data_vardof_hard/pose_UR5e.npz ]  || run pose UR5e
[ -s data_vardof_hard/pos_UR5e.npz ]   || run pos  UR5e

echo "[vdh] done $(date -u +%H:%M:%S)"
grep -h "demo success" vdh_*.log 2>/dev/null
echo "VARDOF_HARD_COLLECT_DONE=ok" > vardof_hard_collect.status
echo "[vdh] finished $(date -u +%H:%M:%S)"
