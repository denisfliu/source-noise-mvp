#!/bin/bash
# kill any prior worker (pattern not present in this script's own cmdline)
pkill -9 -f rung2_transfer.py 2>/dev/null
sleep 1
cd ~/code/source-noise-mvp/experiments/rung2
nohup ~/.local/bin/uv run --with autograd --with numpy --python 3.11 python rung2_transfer.py > /tmp/rung2_transfer.log 2>&1 &
disown
echo "launched pid $!"
