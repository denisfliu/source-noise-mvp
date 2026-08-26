#!/bin/bash
# scratch-on-gate_nav3 control: wait for training, then plain-serve six cells.
set -u
RUN=/home/dfliu/ctxrun
CK=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch3/4999
for k in $(seq 1 480); do [ -d "$CK/params" ] && break; sleep 60; done
[ -d "$CK/params" ] || { echo TRAIN_TIMEOUT > $RUN/ev6_scr3.done; exit 1; }
sleep 60
rm -rf /home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch3/*/train_state
bash /home/dfliu/code/source-noise-mvp/scripts/run_sixcell_plain.sh scr3 $CK 9030 10
