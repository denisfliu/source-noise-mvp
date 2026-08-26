#!/bin/bash
# Collect obstacle-reach demos on the real robosuite arm (writes data_obst/<ARM>.npz).
# Env pins REQUIRED: robosuite 1.4.1 + mujoco 2.3.7 (3.x breaks robosuite 1.4.1).
cd ~/code/source-noise-mvp/experiments/rung2
MUJOCO_GL=egl nohup ~/.local/bin/uv run --with 'robosuite==1.4.1' --with 'mujoco==2.3.7' \
  --with numpy --python 3.11 python collect_obstacle.py > /tmp/rung2_obst.log 2>&1 &
disown
echo "launched obstacle collect pid $! -> /tmp/rung2_obst.log"
