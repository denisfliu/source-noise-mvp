#!/bin/bash
# Post for run_vproj_compare.sh: realism ledger on the sketch window per cell (all arms side by side) and the cloud page.
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1"
OUT=$RUN/vproj_realism.txt; rm -f $OUT
cd $RD
{ echo "== orbit (right scene): realism on the sketch-active window"
  $EV $VENVPY realism.py --segment sketch --sketch $RD/sketch_orbit.json --out $RUN/vproj_realism_orbit.json \
     --adhoc "ours (real-only pin)=app_realonly_orbit" --adhoc "SDEdit t0 0.5 (scratch real)=vp_sde05_orbit" \
     --adhoc "inject (scratch real)=vp_inject_orbit" --adhoc "inject + v-projection (scratch real)=vp_vproj_orbit"
  echo; echo "== fig8 (right scene)"
  $EV $VENVPY realism.py --segment sketch --sketch $RD/sketch_fig8.json --out $RUN/vproj_realism_fig8.json \
     --adhoc "ours (real-only pin)=app_realonly_fig8" --adhoc "SDEdit t0 0.5 (scratch real)=vp_sde05_fig8" \
     --adhoc "inject (scratch real)=vp_inject_fig8" --adhoc "inject + v-projection (scratch real)=vp_vproj_fig8"
  echo; echo "== hand-drawn compound (left+center)"
  $EV $VENVPY realism.py --segment sketch --sketch $RD/sketch_cmpl_denis.json --out $RUN/vproj_realism_cmpl.json \
     --adhoc "ours (real-only pin)=vp_ours_cmpl" --adhoc "SDEdit t0 0.5 (scratch real)=vp_sde05_cmpl" \
     --adhoc "inject (scratch real)=vp_inject_cmpl" --adhoc "inject + v-projection (scratch real)=vp_vproj_cmpl"
} > $OUT 2>&1
/home/dfliu/miniforge3/bin/python3 viz/build_traj_page.py --no-judge --out vproj_compare.html --title "Sketch Injection Three Ways" \
  --section "Orbit, right scene|right|$RD/sketch_orbit.json|ours (real-only pin)=app_realonly_orbit,SDEdit t0 0.5=vp_sde05_orbit,inject=vp_inject_orbit,inject + v-projection=vp_vproj_orbit" \
  --section "Figure-eight, right scene|right|$RD/sketch_fig8.json|ours (real-only pin)=app_realonly_fig8,SDEdit t0 0.5=vp_sde05_fig8,inject=vp_inject_fig8,inject + v-projection=vp_vproj_fig8" \
  --section "Hand-drawn compound, left then center|left_and_center|$RD/sketch_cmpl_denis.json|ours (real-only pin)=vp_ours_cmpl,SDEdit t0 0.5=vp_sde05_cmpl,inject=vp_inject_cmpl,inject + v-projection=vp_vproj_cmpl" \
  --note "All arms use the real-only checkpoints: the pin (gate_pin_joint_realonly) for ours, pi0 fine-tuned on the same 100 real demos (gate_scratch_real) for the three injection strategies. Same sketches, same simulator, 5 trials each." >> $OUT 2>&1
tail -5 $OUT
