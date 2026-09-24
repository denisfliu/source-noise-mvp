#!/bin/bash
# pi0 25-step cell from n=20 to n=100 with a distinct noise seed per batch of ten (t21-30 -> seed 2 ... t91-100 -> seed 9).
set -u; cd /home/dfliu/code/source-noise-mvp
for b in 2 3 4 5 6 7 8 9; do T0=$((b * 10 + 1)); SEED=$b TRIAL0=$T0 TRIALS=10 PORT=9021 bash scripts/run_scrreal_apc25.sh 25; echo "BATCH_DONE pi0 $b"; done
echo PI0_100_DONE
