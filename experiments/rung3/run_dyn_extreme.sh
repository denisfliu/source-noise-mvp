#!/bin/bash
# Fair broken-ACH test: aggressive but LATENCY-FREE dynamics gaps (large gain/damping
# outside set-A's kp130-200 / damp0.85-1.1 range). Latency destroys feasibility on this
# fixed-horizon tracking task (lat3->0.06, lat5->0.00), so use gain/damping instead and
# check whether a FEASIBLE dynamics gap can break the achieved-pin's coordinate invariance.
# GPU 0 ONLY, variants run SEQUENTIALLY (the other GPU is reserved).
# Launch: setsid bash run_dyn_extreme.sh < /dev/null > dyn_extreme.log 2>&1 & disown
set -u
cd ~/code/source-noise-mvp/experiments/rung3
UVR="$HOME/.local/bin/uv run --with robosuite==1.4.1 --with mujoco==2.3.7 --with numpy --python 3.11 python"
echo "[ext] start $(date -u +%H:%M:%S)"

collect() {  # name kp damp lat
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    SNMVP_VNAME=$1 SNMVP_KP=$2 SNMVP_DAMP=$3 SNMVP_LAT=$4 $UVR collect_dyn.py > dyn_$1.log 2>&1
}

# stiff + underdamped (ringing transient, large realization gap, no latency -> should still reach)
[ -s data_dyn/stiff.npz ] || collect stiff 330 0.6 0
# soft + overdamped (sluggish; may undershoot -> feasibility to be measured)
[ -s data_dyn/soft.npz ]  || collect soft 95 1.45 0

echo "[ext] done $(date -u +%H:%M:%S)"
grep -h "demo success" dyn_stiff.log dyn_soft.log 2>/dev/null
echo "DYN_EXTREME_DONE=ok" > dyn_extreme.status
echo "[ext] finished $(date -u +%H:%M:%S)"
