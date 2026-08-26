#!/bin/bash
# Generative head x multi-horizon basis (the combined bet: routes from mh16's code, valid
# full-magnitude commands from sampling). Waits for the seed-7 replication to release GPU1.
set -u
RUN=/home/ubuntu/ctxrun
for k in $(seq 1 720); do [ -f $RUN/ctr_b2lam03s7.done ] && break; sleep 60; done
bash /home/ubuntu/run_joint_arm3.sh gen16 1 8931 5000 1000000 10 \
  /home/ubuntu/code/source-noise-mvp/experiments/rung3/pin_U_mh16.npy \
  "SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GEN=1" >> $RUN/gen16_chain.log 2>&1
