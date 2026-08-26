#!/bin/bash
# Self-contained variable-DOF transfer pipeline. Runs collection (if needed) and
# all transfer tests on the box, independent of any SSH session. Launch with:
#   setsid bash run_vardof_pipeline.sh < /dev/null > vardof_pipeline.log 2>&1 &
set -u
cd ~/code/source-noise-mvp/experiments/rung3
UVR="$HOME/.local/bin/uv run --with robosuite==1.4.1 --with mujoco==2.3.7 --with numpy --python 3.11 python"
VENV=../../.venv/bin/python
echo "[pipeline] start $(date -u +%H:%M:%S)"

# 1. collect three-channel (OSC_POSITION) data on Panda and UR5e if missing
if [ ! -s data_pos3/Panda.npz ] || [ ! -s data_pos3/UR5e.npz ]; then
  echo "[pipeline] collecting C=3 data"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 SNMVP_ARM=Panda $UVR collect_pos3.py > pos3_Panda.log 2>&1 &
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 SNMVP_ARM=UR5e  $UVR collect_pos3.py > pos3_UR5e.log 2>&1 &
  wait
fi
echo "[pipeline] collection done $(date -u +%H:%M:%S)"
grep -h "pos3 success" pos3_Panda.log pos3_UR5e.log 2>/dev/null

# 2. variable-DOF transfer tests (each writes vardof_<held>_result.json)
echo "[pipeline] 6->3: held-out Panda:3"
SNMVP_SETA=IIWA:6,UR5e:6,Jaco:6 SNMVP_HELD=Panda:3 $VENV vardof_transfer.py > vardof_Panda3.log 2>&1
echo "[pipeline] 6->3: held-out UR5e:3"
SNMVP_SETA=Panda:6,IIWA:6,Jaco:6 SNMVP_HELD=UR5e:3 $VENV vardof_transfer.py > vardof_UR5e3.log 2>&1
echo "[pipeline] 3->6: held-out Jaco:6"
SNMVP_SETA=Panda:3,UR5e:3 SNMVP_HELD=Jaco:6 $VENV vardof_transfer.py > vardof_Jaco6.log 2>&1

echo "ALL_DONE $(date -u +%H:%M:%S)" > vardof_pipeline.status
echo "[pipeline] finished"
