#!/bin/bash
set -u
RD=~/code/source-noise-mvp/experiments/rung3
until [ -d ~/code/openpi/checkpoints/pi0_libero_shared/snmvp_src_pin_vlm_soft/4999 ]; do sleep 60; done
echo "SOFT_BASE_READY $(date -u +%H:%M:%S)" >> $RD/soft_cl.status
setsid bash $RD/run_hh_soft.sh 0 15 </dev/null >$RD/hh_soft_driver.log 2>&1
echo "SOFT_CL_DONE $(date -u +%H:%M:%S)" >> $RD/soft_cl.status
