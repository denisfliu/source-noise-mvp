#!/bin/bash
# Collect 6-DOF pose-reach-around-obstacle demos on the real robosuite arm
# (writes data_pose6d/<ARM>.npz). Env pins: robosuite 1.4.1 + mujoco 2.3.7.
# GPU 1 by default (EGL device tied to CUDA_VISIBLE_DEVICES).
cd ~/code/source-noise-mvp/experiments/rung3
GPU="${SNMVP_GPU:-1}"
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU CUDA_VISIBLE_DEVICES=$GPU \
  nohup ~/.local/bin/uv run --with 'robosuite==1.4.1' --with 'mujoco==2.3.7' \
  --with numpy --python 3.11 python collect_pose6d.py > /tmp/rung3_pose6d.log 2>&1 &
disown
echo "launched pose6d collect pid $! -> /tmp/rung3_pose6d.log"
