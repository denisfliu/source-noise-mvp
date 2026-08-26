#!/bin/bash
# B-loss arm (Denis, 2026-08-12): b2lam03's exact recipe with ONLY the head loss changed to the
# scale-invariant direction+log-magnitude form (SNMVP_HEAD_LOGMAG=1). Same basis, same lam, same
# steps -> single-variable comparison. Chained last on GPU1 (after b2long's center add-on).
# Watch arm_b2logmag_train.log for stability; the readout gate still measures plain c-R2.
set -u
RUN=/home/ubuntu/ctxrun
for k in $(seq 1 900); do [ -f $RUN/ctr_b2long.done ] && break; sleep 60; done
bash /home/ubuntu/run_joint_arm3.sh b2logmag 1 8921 5000 1000000 10 \
  /home/ubuntu/code/source-noise-mvp/experiments/rung3/pin_U_gate_rrr_k5.npy \
  "SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_LOGMAG=1" \
  >> $RUN/b2logmag_chain.log 2>&1
