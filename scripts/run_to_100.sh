#!/bin/bash
# Take the 25-step real-only cells (ours and pi0, left and right) from n=20 to n=100 (Denis, 2026-09-24): eight more
# batches of ten per arm, one server process per batch with its own residual-noise seed (SEED = batch index, so
# t21-30 -> seed 2 ... t91-100 -> seed 9; a restarted pin server at the same seed would replay the cell).
set -u
cd /home/dfliu/code/source-noise-mvp
for b in 2 3 4 5 6 7 8 9; do
  T0=$((b * 10 + 1))
  SEED=$b TRIAL0=$T0 TRIALS=10 PORT=9020 bash scripts/run_realonly_apc25.sh realonly 25; echo "BATCH_DONE ours $b"
done
for b in 2 3 4 5 6 7 8 9; do
  T0=$((b * 10 + 1))
  TRIAL0=$T0 TRIALS=10 PORT=9021 bash scripts/run_scrreal_apc25.sh 25; echo "BATCH_DONE pi0 $b"
done
echo CHAIN100_DONE
