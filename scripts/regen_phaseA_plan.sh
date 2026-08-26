#!/bin/bash
# Synth regeneration phase A (2026-08-23): plan 70 variants per fixed course with the
# real-matched start distribution, score every plan (posthoc judge + gate clearance), keep the
# FIRST 50 passing per task into runs/regen1/kept/<course>/. Same per-task counts as the
# original dataset by construction. Phase B renders the kept NPZs.
set -u
FAL=/home/dfliu/code/falsify
RD=/home/dfliu/code/source-noise-mvp/experiments/rung3
VENVPY=/home/dfliu/code/openpi/.venv/bin/python
TV=/home/dfliu/code/tv/bin/python
OUT=$FAL/runs/regen1
START='--start-mean=-0.20,0.05,-0.06 --start-jitter=0.15,0.10,0.12,0.06'
cd $FAL && source tools/env.sh
mkdir -p $OUT
for spec in through_left_gate:left_gate:left through_right_gate:right_gate:right \
            through_center_gate_from_left:center_gate:center_from_left \
            through_center_gate_from_right:center_gate:center_from_right; do
  CO=${spec%%:*}; rest=${spec#*:}; SC=${rest%%:*}; SIDE=${rest##*:}
  PLAN=$OUT/planned/$CO; KEPT=$OUT/kept/$CO; NP=$OUT/np/$CO
  rm -rf $PLAN $KEPT $NP; mkdir -p $PLAN $KEPT $NP
  PYTHONPATH=src .venv/bin/python scripts/dataset/plan_course_variants.py \
    --course configs/courses/$CO.yaml --scene configs/scenes/$SC.yaml --planner mpc \
    $START --out-dir $PLAN > $OUT/plan_$CO.log 2>&1
  python3 - "$PLAN" "$NP" << 'PYEOF'
import sys, glob, os
import numpy as np
plan, npd = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(f"{plan}/*.npz")):
    P = np.load(f, allow_pickle=True)["positions_ned"] * np.array([1.0, -1.0, -1.0])
    np.save(f"{npd}/{os.path.basename(f).replace('.npz','')}.npy", P.astype(np.float32))
PYEOF
  # score all planned trajectories; keep first 50 that pass BOTH judge and clearance
  env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu \
    CUDA_VISIBLE_DEVICES=-1 $VENVPY $RD/gate_success.py --traj $NP/*.npy --side $SIDE \
    > $OUT/judge_$CO.txt 2>&1
  CLSCENE=$SC; case $SC in center_gate) CLSCENE=center;; left_gate) CLSCENE=left;; right_gate) CLSCENE=right;; esac
  $TV $RD/gate_clearance.py --scene $CLSCENE --traj $NP/*.npy > $OUT/clear_$CO.txt 2>&1
  python3 - "$OUT" "$CO" "$PLAN" "$KEPT" << 'PYEOF'
import sys, os, shutil, re
out, co, plan, kept = sys.argv[1:5]
ok_j = {m.group(1) for m in re.finditer(r"(\S+)\.npy\s.*SUCCESS=True", open(f"{out}/judge_{co}.txt").read())}
ok_c = {m.group(1) for m in re.finditer(r"(\S+)\.npy\s.*CLEAN=True", open(f"{out}/clear_{co}.txt").read())}
names = sorted(n for n in ok_j & ok_c)
kept_n = 0
for n in names:
    src = f"{plan}/{n}.npz"
    if os.path.exists(src) and kept_n < 50:
        shutil.copy(src, f"{kept}/{n}.npz")
        kept_n += 1
print(f"[{co}] judge-pass {len(ok_j)}  clean {len(ok_c)}  both {len(names)}  kept {kept_n}")
PYEOF
done
echo PHASE_A_DONE
