#!/bin/bash
# SDEdit vs pin discriminators (2026-09-03, Denis: "test those").
#  A. disturbance rejection: hand-drawn L->C, 0.4 m lateral kick at step 100 (the 2026-08-28
#     carrot/kick protocol), pin (xswap s42) vs SDEdit (scratch3), carrot 0 / 20.
#  B. badly drawn sketches: the original 4-click L->C (sketch clearance 0.07 m) and 4-click R->C
#     (0.04 m) through SDEdit (pin numbers exist: xsk42/xsks7_cmpl_min4, xsk42_cmpr_min4);
#     the hand-drawn L->C at 2.5x pace, pin vs SDEdit.
#  C. real-frame probe (offline) for SDEdit at t0 in {0.3,0.5,0.7,1.0}.
set -u
RUN=/home/dfliu/ctxrun
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
HFB=/home/dfliu/hf_bundle/gate-drone-pi0
EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src XLA_PYTHON_CLIENT_PREALLOCATE=false"
GPU="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 CUDA_VISIBLE_DEVICES=0"
U=$RD/pin_U_mh16.npy
PINENV="SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U=$U SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1 SNMVP_SIGMA_MAP=$RD/sigma_map_xswap.json"
CK_PIN=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_pin_joint_xswap/4999
CK_SCR=/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3/gate_scratch3/4999
CMP_L="go through the gate on the left, then through the center gate and hover over the stuffed animal"
CMP_R="go through the gate on the right, then through the center gate and hover over the stuffed animal"
PORT=9150
OUT=$RUN/sdedit2_scores.txt
rm -f $RUN/sdedit2.done $OUT
cd $RD
killport () { for p in $(pgrep -f "port $PORT"); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
cell () { # tag mode(pin|sde:T0) sketch side scene prompt kick("" or step:dx,dy,dz)
  local TAG=$1 MODE=$2 SK=$3 SIDE=$4 SCENE=$5 PROMPT=$6 KK=$7
  killport
  if [ "$MODE" = pin ]; then
    setsid $EV $PINENV SNMVP_PIN_PROMPT=$RD/$SK CLOG=$RUN/clog_${TAG}.npy $GPU \
      $VENVPY $RD/serve_gate_pin_joint.py --ckpt $CK_PIN --config pi0_gate --norm $HFB/assets/gate_nav \
      --pin-u $U --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  else
    setsid $EV SNMVP_ZERO_PAD_ACTIONS=1 $GPU \
      $VENVPY $RD/serve_gate_sdedit.py --ckpt $CK_SCR --config pi0_gate --norm $HFB/assets/gate_nav \
      --sketch $RD/$SK --t0 ${MODE#sde:} --port $PORT >> $RUN/sv_${TAG}.log 2>&1 </dev/null & disown
  fi
  for k in $(seq 1 150); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done
  ss -ltn | grep -q ":$PORT " || { echo "SERVER_TIMEOUT $TAG" >> $OUT; return 1; }
  env CUDA_VISIBLE_DEVICES=0 PORT=$PORT SIDE=$SIDE SCENE=$SCENE NCH=14 APC=50 TRIALS=5 VIDEO=0 \
    ${KK:+KICK=$KK} PROMPT="$PROMPT" TRAJ=$RUN/traj_${TAG}_{t}.npy $TV $RD/gate_rollout_batch.py \
    > $RUN/roll_${TAG}.log 2>&1
  killport
  { echo "== sdedit2: $TAG (mode=$MODE sketch=$SK kick=${KK:-none})"
    $EV JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $RUN/traj_${TAG}_*.npy --side $SCENE
    $TV $RD/gate_clearance.py --scene $SCENE --traj $RUN/traj_${TAG}_*.npy
    $TV $RD/sketch_track.py --sketch $RD/$SK --traj $RUN/traj_${TAG}_*.npy
    [ -n "$KK" ] && { echo "-- post-kick rejoin (from step 100)"; $TV $RD/sketch_track.py --sketch $RD/$SK --from-step 100 --traj $RUN/traj_${TAG}_*.npy; }
  } >> $OUT 2>&1
}
K="100:0,-0.4,0"
# A. kick
cell kx_pin_c0     pin     sketch_cmpl_denis.json     left left_and_center "$CMP_L" "$K"
cell kx_pin_c20    pin     sketch_cmpl_denis_c20.json left left_and_center "$CMP_L" "$K"
cell kx_sde03_c0   sde:0.3 sketch_cmpl_denis.json     left left_and_center "$CMP_L" "$K"
cell kx_sde03_c20  sde:0.3 sketch_cmpl_denis_c20.json left left_and_center "$CMP_L" "$K"
cell kx_sde05_c20  sde:0.5 sketch_cmpl_denis_c20.json left left_and_center "$CMP_L" "$K"
# B. bad sketches
cell bs_sde03_m4L  sde:0.3 sketch_cmpl_min4.json      left  left_and_center  "$CMP_L" ""
cell bs_sde05_m4L  sde:0.5 sketch_cmpl_min4.json      left  left_and_center  "$CMP_L" ""
cell bs_sde03_m4R  sde:0.3 sketch_cmpr_min4.json      right right_and_center "$CMP_R" ""
cell bs_sde05_m4R  sde:0.5 sketch_cmpr_min4.json      right right_and_center "$CMP_R" ""
cell fs_pin        pin     sketch_cmpl_denis_fast.json left left_and_center "$CMP_L" ""
cell fs_sde03      sde:0.3 sketch_cmpl_denis_fast.json left left_and_center "$CMP_L" ""
# C. real-frame probe
killport
{ echo "== sdedit2: real-frame probe (scratch3, SDEdit guide = frame's own demo chunk)"
  $EV $GPU $VENVPY $RD/sdedit_real_probe.py --ckpt $CK_SCR --t0 0.3 0.5 0.7 1.0 --frames 60 --out $RUN/sdedit_real_probe.json
} >> $OUT 2>&1
echo DONE > $RUN/sdedit2.done
