#!/bin/bash
# Collect two-obstacle SLALOM demos on the real robosuite arm (writes data_slalom/<ARM>.npz).
# Env pins REQUIRED: robosuite 1.4.1 + mujoco 2.3.7 (3.x breaks robosuite 1.4.1).
cd ~/code/source-noise-mvp/experiments/rung2
MUJOCO_GL=egl nohup ~/.local/bin/uv run --with 'robosuite==1.4.1' --with 'mujoco==2.3.7' \
  --with numpy --python 3.11 python collect_slalom.py > /tmp/rung2_slalom.log 2>&1 &
disown
echo "launched slalom collect pid $! -> /tmp/rung2_slalom.log"
