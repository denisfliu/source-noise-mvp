#!/bin/bash
# Synth regeneration phase B (2026-08-23): render the kept 50 trajectories per course into
# per-episode parquets (falsify exporter, RGB pinhole convention). Waits for phase A's marker.
# ~1-2 min/episode on the 4090 -> ~4-7 h for 200. Verifies parquet EXISTENCE per episode (the
# exporter's summary line lies on failure — 2026-08-22 finding).
set -u
FAL=/home/dfliu/code/falsify
OUT=$FAL/runs/regen1
for k in $(seq 1 120); do grep -qa PHASE_A_DONE /home/dfliu/ctxrun/regen_phaseA.log 2>/dev/null && break; sleep 60; done
grep -qa PHASE_A_DONE /home/dfliu/ctxrun/regen_phaseA.log || { echo PHASE_A_TIMEOUT; exit 1; }
cd $FAL && source tools/env.sh
export PATH=$FAL/.venv/bin:$PATH
for spec in through_left_gate:left_gate through_right_gate:right_gate \
            through_center_gate_from_left:center_gate through_center_gate_from_right:center_gate; do
  CO=${spec%%:*}; SC=${spec##*:}
  STAGE=$OUT/staging/$CO
  mkdir -p $STAGE
  PYTHONPATH=src:external/FiGS/src:external/splatnav .venv/bin/python -m falsify.cli.export_training_data \
    --trajectories-dir $OUT/kept/$CO \
    --scene configs/scenes/$SC.yaml \
    --frame configs/frames/carl_dual.yaml \
    --embodiment configs/embodiments/carl_dual_mocap.yaml \
    --out $STAGE >> /home/dfliu/ctxrun/regen_render_$CO.log 2>&1
  N=$(find $STAGE -name "*.parquet" | wc -l)
  echo "[render] $CO: $N parquets"
done
echo PHASE_B_DONE
