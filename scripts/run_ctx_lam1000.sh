#!/bin/bash
# Retest contextualized VLM-c closed-loop with the heavier-ridge map (lam=1000, maxW 0.31).
export PATH=/tmp/tv/bin:/usr/local/cuda-12.8/bin:$PATH; export CUDA_HOME=/usr/local/cuda-12.8
RUN=/home/ubuntu/ctxrun; rm -f $RUN/lam1000.done
PY=/home/ubuntu/code/openpi/.venv/bin/python; HFB=/home/ubuntu/hf_bundle/gate-drone-pi0
RD=/home/ubuntu/code/source-noise-mvp/experiments/rung3; RRRCK=/home/ubuntu/code/openpi/checkpoints/pi0_gate/gate_both_pin_rrr/4999
EV="env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false"
WC=/tmp/vlmc_ridge_ctx_lam1000.npz
echo "[1] two servers (lam1000 ctx map)" > $RUN/chain_lam1000.log
pkill -9 -f serve_gate 2>/dev/null; sleep 4; rm -f $RUN/svA1k.log $RUN/svB1k.log
setsid $EV CUDA_VISIBLE_DEVICES=0 $PY /tmp/serve_gate_pin_vlmc.py --ckpt $RRRCK --norm $HFB/assets/gate_nav --pin-u $RD/pin_U_gate_rrr_k5.npy --wc $WC --port 8796 >> $RUN/svA1k.log 2>&1 </dev/null & disown
setsid $EV CUDA_VISIBLE_DEVICES=1 $PY /tmp/serve_gate_pin_vlmc.py --ckpt $RRRCK --norm $HFB/assets/gate_nav --pin-u $RD/pin_U_gate_rrr_k5.npy --wc $WC --port 8797 >> $RUN/svB1k.log 2>&1 </dev/null & disown
for k in $(seq 1 120); do (grep -qa "ready on ws" $RUN/svA1k.log 2>/dev/null && grep -qa "ready on ws" $RUN/svB1k.log 2>/dev/null) && break; sleep 3; done
echo "[2] parallel left+right rollouts" >> $RUN/chain_lam1000.log
CUDA_VISIBLE_DEVICES=0 PORT=8796 SIDE=left  SCENE=left  NCH=40 OUT=$RUN/overlay_left_1k.mp4  TRAJ=$RUN/traj_ctx_left_1k.npy  /tmp/tv/bin/python /tmp/gate_video_overlay.py > $RUN/score_left_1k.txt  2>&1 &
CUDA_VISIBLE_DEVICES=1 PORT=8797 SIDE=right SCENE=right NCH=40 OUT=$RUN/overlay_right_1k.mp4 TRAJ=$RUN/traj_ctx_right_1k.npy /tmp/tv/bin/python /tmp/gate_video_overlay.py > $RUN/score_right_1k.txt 2>&1 &
wait
pkill -9 -f serve_gate 2>/dev/null
echo DONE > $RUN/lam1000.done
