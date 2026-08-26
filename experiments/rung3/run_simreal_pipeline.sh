#!/bin/bash
# Self-contained sim-to-real transfer pipeline (fixed action interface, varying
# dynamics). Launch detached with:
#   setsid bash run_simreal_pipeline.sh < /dev/null > simreal_pipeline.log 2>&1 & disown
set -u
cd ~/code/source-noise-mvp/experiments/rung3
UVR="$HOME/.local/bin/uv run --with robosuite==1.4.1 --with mujoco==2.3.7 --with numpy --python 3.11 python"
VENV=../../.venv/bin/python
echo "[simreal] start $(date -u +%H:%M:%S)"

collect() {  # name kp damp lat gpu
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$5 CUDA_VISIBLE_DEVICES=$5 \
    SNMVP_VNAME=$1 SNMVP_KP=$2 SNMVP_DAMP=$3 SNMVP_LAT=$4 $UVR collect_dyn.py > dyn_$1.log 2>&1
}

# simulated training variants (feasible, moderate dynamics range) and one held-out
# variant with gain and damping outside that range (verified feasible, ceiling ~0.89)
[ -s data_dyn/sim1.npz ] || collect sim1 150 1.0 0 0 &
[ -s data_dyn/sim2.npz ] || collect sim2 130 1.1 0 1 &
wait
[ -s data_dyn/sim3.npz ] || collect sim3 200 0.85 1 0 &
[ -s data_dyn/real.npz ] || collect real 250 0.75 0 1 &
wait
echo "[simreal] collection done $(date -u +%H:%M:%S)"
grep -h "demo success" dyn_sim1.log dyn_sim2.log dyn_sim3.log dyn_real.log 2>/dev/null

SNMVP_SETA=sim1,sim2,sim3 SNMVP_HELD=real $VENV simreal_transfer.py > simreal_transfer.log 2>&1
echo "ALL_DONE $(date -u +%H:%M:%S)" > simreal_pipeline.status
echo "[simreal] finished $(date -u +%H:%M:%S)"
