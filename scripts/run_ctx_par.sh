#!/bin/bash
export PATH=/tmp/tv/bin:/usr/local/cuda-12.8/bin:$PATH; export CUDA_HOME=/usr/local/cuda-12.8
RUN=/home/ubuntu/ctxrun; mkdir -p $RUN; rm -f $RUN/all.done $RUN/Xshard_*.npy /tmp/vlmc_ridge_ctx.npz
PY=/home/ubuntu/code/openpi/.venv/bin/python; HFB=/home/ubuntu/hf_bundle/gate-drone-pi0
RD=/home/ubuntu/code/source-noise-mvp/experiments/rung3; RRRCK=/home/ubuntu/code/openpi/checkpoints/pi0_gate/gate_both_pin_rrr/4999
EV="env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false"
echo "[1] parallel shard extraction (2 GPUs)" > $RUN/chain.log
$EV CUDA_VISIBLE_DEVICES=0 MODE=extract SHARD_K=0 SHARD_N=2 RUN=$RUN $PY /tmp/extract_ctx2.py > $RUN/shard0.log 2>&1 &
$EV CUDA_VISIBLE_DEVICES=1 MODE=extract SHARD_K=1 SHARD_N=2 RUN=$RUN $PY /tmp/extract_ctx2.py > $RUN/shard1.log 2>&1 &
wait
if [ ! -f $RUN/Xshard_0.npy ] || [ ! -f $RUN/Xshard_1.npy ]; then echo "SHARD_FAILED" > $RUN/all.done; exit 1; fi
echo "[2] build ridge map" >> $RUN/chain.log
$EV CUDA_VISIBLE_DEVICES=1 MODE=build SHARD_N=2 RUN=$RUN $PY /tmp/extract_ctx2.py > $RUN/ctxbuild.log 2>&1
if [ ! -f /tmp/vlmc_ridge_ctx.npz ]; then echo "BUILD_FAILED" > $RUN/all.done; exit 1; fi
echo "[3] two servers" >> $RUN/chain.log
pkill -9 -f serve_gate 2>/dev/null; sleep 4; rm -f $RUN/svA.log $RUN/svB.log
setsid $EV CUDA_VISIBLE_DEVICES=0 $PY /tmp/serve_gate_pin_vlmc.py --ckpt $RRRCK --norm $HFB/assets/gate_nav --pin-u $RD/pin_U_gate_rrr_k5.npy --wc /tmp/vlmc_ridge_ctx.npz --port 8796 >> $RUN/svA.log 2>&1 </dev/null & disown
setsid $EV CUDA_VISIBLE_DEVICES=1 $PY /tmp/serve_gate_pin_vlmc.py --ckpt $RRRCK --norm $HFB/assets/gate_nav --pin-u $RD/pin_U_gate_rrr_k5.npy --wc /tmp/vlmc_ridge_ctx.npz --port 8797 >> $RUN/svB.log 2>&1 </dev/null & disown
for k in $(seq 1 120); do (grep -qa "ready on ws" $RUN/svA.log 2>/dev/null && grep -qa "ready on ws" $RUN/svB.log 2>/dev/null) && break; sleep 3; done
echo "[4] parallel left+right overlay+score rollouts" >> $RUN/chain.log
CUDA_VISIBLE_DEVICES=0 PORT=8796 SIDE=left  SCENE=left  NCH=40 OUT=$RUN/overlay_left.mp4  TRAJ=$RUN/traj_ctx_left.npy  /tmp/tv/bin/python /tmp/gate_video_overlay.py > $RUN/score_left.txt  2>&1 &
CUDA_VISIBLE_DEVICES=1 PORT=8797 SIDE=right SCENE=right NCH=40 OUT=$RUN/overlay_right.mp4 TRAJ=$RUN/traj_ctx_right.npy /tmp/tv/bin/python /tmp/gate_video_overlay.py > $RUN/score_right.txt 2>&1 &
wait
pkill -9 -f serve_gate 2>/dev/null
echo DONE > $RUN/all.done
