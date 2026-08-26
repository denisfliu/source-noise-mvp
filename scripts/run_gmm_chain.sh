#!/bin/bash
# GMM/MDN command-head arm (toy_cmdhead 2026-08-19 -> box arm): b2lam03's exact recipe with ONLY
# the head swapped to the mixture head (SNMVP_HEAD_GMM=1, M=4, FiLM information diet, NLL loss,
# argmax+pi-hysteresis serve). Single-variable vs the ctl MSE twin on the same rebuilt basis.
# Waits for ctl's chain to release the GPU (done marker win or lose: a gate failure still frees it).
set -u
RUN=/home/dfliu/ctxrun
# ctl's train+gate passed but its first eval died on missing tv-venv packages (2026-08-20 01:00);
# the re-fly runs as run_sixcell_eval_local.sh -> gate on ITS marker, not arm_ctl.done (stale).
for k in $(seq 1 1440); do [ -f $RUN/ev6_ctl.done ] && break; sleep 60; done
[ -f $RUN/ev6_ctl.done ] || { echo WAIT_TIMEOUT > $RUN/arm_gmm.done; exit 1; }
bash /home/dfliu/code/source-noise-mvp/scripts/run_joint_arm_local.sh gmm 8950 5000 1000000 10 \
  /home/dfliu/code/source-noise-mvp/experiments/rung3/pin_U_gate_rrr_k5.npy \
  "SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1" \
  >> $RUN/gmm_chain.log 2>&1
