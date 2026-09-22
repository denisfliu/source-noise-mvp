#!/bin/bash
# The evaluation suite (docs/AGENT_EVAL_PLAN.md): N trials of every task in both arms, ours then waypoints, SEQUENTIAL:
# two servers plus two renderers do not fit on the 24 GB card (the second renderer OOMs at scene load, 2026-09-23).
# Results accumulate in
# experiments/rung3/agent_eval/<task>_<arm>[_model].jsonl.
#   bash scripts/run_agent_eval_suite.sh <ntrials> [first_trial=1] [model=sonnet] [tasks="mannequin double orbit left_mannequin"]
set -u
N=${1:-5}; T0=${2:-1}; MODEL=${3:-sonnet}; TASKS=${4:-"mannequin double orbit left_mannequin"}
ROOT=/home/dfliu/code/source-noise-mvp; RUN=/home/dfliu/ctxrun
chain () {  # arm port
  for t in $TASKS; do MODEL=$MODEL PORT=$2 bash $ROOT/scripts/run_agent_matrix.sh $t $1 $N $T0; done
  echo "CHAIN_DONE $1"
}
chain ours 9160 > $RUN/suite_${MODEL}_ours.log 2>&1
chain waypoints 9160 > $RUN/suite_${MODEL}_waypoints.log 2>&1
echo SUITE_DONE
