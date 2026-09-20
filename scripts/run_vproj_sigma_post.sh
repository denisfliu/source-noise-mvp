#!/bin/bash
# Realism ledger + page for the partial-projection sweep, alongside the 2026-09-20 arms.
set -u
RUN=/home/dfliu/ctxrun; RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python; EV="env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1"
OUT=$RUN/vproj_sigma_realism.txt; rm -f $OUT
cd $RD
arms () { local c=$1; local ours=$2; echo --adhoc "ours (real-only pin)=$ours" --adhoc "SDEdit t0 0.5=vp_sde05_$c" --adhoc "v-proj s=0=vp_vproj_$c" --adhoc "v-proj s=0.1=vps01_$c" --adhoc "v-proj s=0.3=vps03_$c" --adhoc "v-proj s=0.5=vps05_$c" --adhoc "inject (s=1)=vp_inject_$c"; }
{ echo "== orbit"; $EV $VENVPY realism.py --segment sketch --sketch $RD/sketch_orbit.json --out $RUN/vprojs_realism_orbit.json $(arms orbit app_realonly_orbit)
  echo "== fig8";  $EV $VENVPY realism.py --segment sketch --sketch $RD/sketch_fig8.json  --out $RUN/vprojs_realism_fig8.json  $(arms fig8 app_realonly_fig8)
  echo "== cmpl";  $EV $VENVPY realism.py --segment sketch --sketch $RD/sketch_cmpl_denis.json --out $RUN/vprojs_realism_cmpl.json $(arms cmpl vp_ours_cmpl)
} > $OUT 2>&1
G () { local c=$1; local ours=$2; echo "ours (real-only pin)=$ours,SDEdit t0 0.5=vp_sde05_$c,v-proj s=0=vp_vproj_$c,v-proj s=0.1=vps01_$c,v-proj s=0.3=vps03_$c,v-proj s=0.5=vps05_$c,inject (s=1)=vp_inject_$c"; }
/home/dfliu/miniforge3/bin/python3 viz/build_traj_page.py --no-judge --out vproj_compare.html --title "Sketch Injection Three Ways" \
  --section "Orbit, right scene|right|$RD/sketch_orbit.json|$(G orbit app_realonly_orbit)" \
  --section "Figure-eight, right scene|right|$RD/sketch_fig8.json|$(G fig8 app_realonly_fig8)" \
  --section "Hand-drawn compound, left then center|left_and_center|$RD/sketch_cmpl_denis.json|$(G cmpl vp_ours_cmpl)" \
  --note "All arms on the real-only checkpoints: the pin (gate_pin_joint_realonly) for ours; pi0 fine-tuned on the same 100 real demos (gate_scratch_real) for SDEdit and for the sketch command written into the source with the flow's velocity projected off the command subspace by a fraction (1 - s) at every Euler step: s = 0 carries the command exactly, s = 1 is naive injection. Same sketches, same simulator, 5 trials each." >> $OUT 2>&1
tail -3 $OUT
