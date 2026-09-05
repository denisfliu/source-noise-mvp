#!/bin/bash
# Score real flights the same way the sim rows are scored (route-clean transit judge + 0.18 m clearance
# against the gate cloud). The gate flight node writes ~/gate_flights/traj_<trial>.npy (N x 3 mocap
# positions at every executed step); copy them to this box and run:
#
#   bash scripts/hw_score.sh <task> traj_a.npy [traj_b.npy ...]
#   task: left | right | center_from_left | center_from_right | compound_left | compound_right
#
# Real-flight caveat: the judge geometry comes from the scene YAMLs registered to the room's mocap frame;
# the pre-flight wand walk (docs/REAL_EXPERIMENT_PLAN.md, R0) is what makes these numbers meaningful.
set -euo pipefail
TASK=${1:?usage: hw_score.sh <task> traj*.npy}; shift
[ $# -gt 0 ] || { echo "no trajectory files given"; exit 2; }
RD=$(cd "$(dirname "$0")/../experiments/rung3" && pwd)
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
case $TASK in
  left)              SIDE=left;              SCENE=left;;
  right)             SIDE=right;             SCENE=right;;
  center_from_left)  SIDE=center_from_left;  SCENE=center;;
  center_from_right) SIDE=center_from_right; SCENE=center;;
  compound_left)     SIDE=left_and_center;   SCENE=left_and_center;;
  compound_right)    SIDE=right_and_center;  SCENE=right_and_center;;
  *) echo "unknown task $TASK"; exit 2;;
esac
echo "== hw_score: task=$TASK judge=$SIDE scene=$SCENE files=$#"
env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
  "$VENVPY" "$RD/gate_success.py" --traj "$@" --side "$SIDE"
"$TV" "$RD/gate_clearance.py" --scene "$SCENE" --traj "$@"
