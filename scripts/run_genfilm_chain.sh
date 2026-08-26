#!/bin/bash
# FiLM-conditioned generative head (Denis-approved): trunk sees (c_t, t) only; state +
# language-token pool + image pool condition via per-layer scale/shift. mh16 basis, lam=0.3 —
# single-variable vs gen16. Waits for gen1det's six-cell pipeline to release GPU0.
set -u
RUN=/home/ubuntu/ctxrun
for k in $(seq 1 720); do [ -f $RUN/ctr_gen1det.done ] && break; sleep 60; done
bash /home/ubuntu/run_joint_arm3.sh genfilm 0 8940 5000 1000000 10 \
  /home/ubuntu/code/source-noise-mvp/experiments/rung3/pin_U_mh16.npy \
  "SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GEN=1 SNMVP_HEAD_FILM=1" \
  >> $RUN/genfilm_chain.log 2>&1
