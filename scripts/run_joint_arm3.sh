#!/bin/bash
# Full six-cell joint arm: train -> readout gate -> left/right (10 trials) -> center CFL/CFR
# (10 trials) -> both compound screens (5 trials). Successor to run_joint_arm2b.sh, which only
# flew 2 of the 4 trained tasks — b2lam03's center failure went unnoticed until Denis asked
# (2026-08-12). Thin orchestrator: arm2 does train+L/R, the center add-on does the rest.
#
#   run_joint_arm3.sh NAME GPU PORT STEPS DECAY TRIALS UPATH "EXTRA_ENV" [TRAIN_CFG]
set -u
NAME=$1; GPU=$2; PORT=$3; STEPS=$4; DECAY=$5; TRIALS=$6; UPATH=$7; EXTRA=${8:-}; CFG=${9:-pi0_gate}; SEED=${10:-42}
bash /home/ubuntu/run_joint_arm2b.sh "$NAME" "$GPU" "$PORT" "$STEPS" "$DECAY" "$TRIALS" "$UPATH" "$EXTRA" "$CFG" "$SEED"
grep -qa DONE /home/ubuntu/ctxrun/arm_$NAME.done || exit 1
CK=/home/ubuntu/code/openpi/checkpoints/$CFG/gate_pin_joint_$NAME/$((STEPS - 1))
bash /home/ubuntu/run_center_addon.sh "$NAME" "$CK" "$GPU" "$((PORT + 1))" "$UPATH" "$EXTRA"
