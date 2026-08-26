#!/bin/bash
# Collect high-latency held-out dynamics variants to create a broken-achieved-pin
# regime (pure actuation-latency shift = FIR delay; deconvolution's ideal case).
# Same OSC_POSE 6-ch interface and gains as sim1 (kp150, d1.0); only latency raised,
# well outside set-A's range (sim1/2/3 latencies 0/0/1). Two candidates to hedge on
# feasibility (a too-high latency drives the ceiling to zero, uninformative).
# Launch detached:
#   setsid bash run_latency_collect.sh < /dev/null > latency_collect.log 2>&1 & disown
set -u
cd ~/code/source-noise-mvp/experiments/rung3
UVR="$HOME/.local/bin/uv run --with robosuite==1.4.1 --with mujoco==2.3.7 --with numpy --python 3.11 python"
echo "[lat] start $(date -u +%H:%M:%S)"

collect() {  # name kp damp lat gpu
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$5 CUDA_VISIBLE_DEVICES=$5 \
    SNMVP_VNAME=$1 SNMVP_KP=$2 SNMVP_DAMP=$3 SNMVP_LAT=$4 $UVR collect_dyn.py > dyn_$1.log 2>&1
}

[ -s data_dyn/lat3.npz ] || collect lat3 150 1.0 3 0 &
[ -s data_dyn/lat5.npz ] || collect lat5 150 1.0 5 1 &
wait
echo "[lat] collection done $(date -u +%H:%M:%S)"
grep -h "demo success" dyn_lat3.log dyn_lat5.log 2>/dev/null
echo "LATENCY_COLLECT_DONE=ok" > latency_collect.status
echo "[lat] finished $(date -u +%H:%M:%S)"
