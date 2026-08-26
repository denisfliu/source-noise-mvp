#!/bin/bash
cd ~/code/source-noise-mvp/experiments/rung2
MUJOCO_GL=egl nohup ~/.local/bin/uv run --with 'robosuite==1.4.1' --with 'mujoco==2.3.7' --with numpy --python 3.11 python collect_demos.py > /tmp/rung2_collect.log 2>&1 &
disown
echo "launched pid $!"
