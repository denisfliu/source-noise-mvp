#!/bin/bash
# Agent-in-the-loop evaluation runner (docs/AGENT_EVAL_PLAN.md, 2026-09-22).
#   bash scripts/run_agent_matrix.sh <task> <arm> <ntrials> [first_trial=1]
#     task: mannequin | double | orbit | left_mannequin      arm: ours | waypoints
# Per trial: render the brief (briefs/common.md + briefs/task_<task>.md with {BACKEND},{DIR},{BUDGET}), launch the
# flight (scripts/run_agent_flight.sh with ARM), wait for decision 0, run a fresh headless Sonnet reviewer
# (`claude -p`), wait for the flight to end, judge the trajectory (agent_judges.py), append one JSON line to
# experiments/rung3/agent_eval/<task>_<arm>.jsonl, and archive decisions + trajectory + reviewer transcript.
set -u
TASK=${1:?task}; ARM=${2:?arm}; N=${3:?ntrials}; T0=${4:-1}
ROOT=/home/dfliu/code/source-noise-mvp; RD=$ROOT/experiments/rung3; RUN=/home/dfliu/ctxrun; OUTD=$RD/agent_eval
MODEL=${MODEL:-sonnet}; MAXTURNS=${MAXTURNS:-120}
# overrides run to completion (2026-09-22 check flights: the 25-step cap left every approach half-flown and the agent
# spent its last decisions closing a gap it believed closed); the reviewer is told so in the brief.
export AGENT_MAXEXEC=${AGENT_MAXEXEC:-50}
case $TASK in
  mannequin)      PROMPT="find the mannequin and hover in front of it"; NCH=14; SY=3.14159;;
  double)         PROMPT="go through the gate on the left, then through the center gate and hover over the stuffed animal"; NCH=14; SY=0;;
  orbit)          PROMPT="fly one full circle around the center gate"; NCH=20; SY=0;;
  left_mannequin) PROMPT="go through the gate on the left, then find the mannequin and hover in front of it"; NCH=24; SY=0;;
  *) echo "unknown task $TASK"; exit 2;;
esac
case $ARM in
  ours) BACKEND="THE BACKEND. A learned policy proposes a plan at every decision; you either APPROVE it or replace it with your own move (OVERRIDE). The policy was trained on exactly two tasks in this room: flying through the gate on the left, and through the gate on the right, to the stuffed animal. It knows nothing else, so approve only when its proposal is the motion you want. Your override is flown by the same policy as a commanded movement: at --sigma 0 it flies your move exactly; at --sigma 0.3 to 0.5 it may bend the move around what its camera sees (use that near gates and furniture)."
    ;;
  waypoints) BACKEND="THE BACKEND. Your move is flown exactly as written, as a smooth path to the point you command, with no correction from the camera. The learned policy that was trained on the left-gate and right-gate tasks is NOT driving here: nothing is proposed and there is nothing to approve, so every decision is an OVERRIDE with a move. --sigma is accepted and has no effect."
    ;;
  *) echo "arm must be ours | waypoints"; exit 2;;
esac
mkdir -p $OUTD $RUN
for i in $(seq $T0 $((T0 + N - 1))); do
  TAG=${TASK}_${ARM}_t$i; AD=$RUN/agent_sim_$TAG
  echo "=== $TAG $(date +%H:%M)"
  ARM=$ARM START="0,0,1.5" STARTYAW=$SY bash $ROOT/scripts/run_agent_flight.sh $TAG left_and_center left "$PROMPT" $NCH 50 > $RUN/launch_$TAG.log 2>&1
  for k in $(seq 1 120); do [ -f $AD/obs/000_view.jpg ] && break; sleep 5; done
  [ -f $AD/obs/000_view.jpg ] || { echo "$TAG: flight never reached decision 0"; continue; }
  python3 - "$RD/briefs/common.md" "$RD/briefs/task_$TASK.md" "$BACKEND" "$AD" "$NCH" > $RUN/brief_$TAG.md <<'PYEOF'
import sys
common, task, backend, d, budget = sys.argv[1:6]
print((open(common).read() + "\n" + open(task).read()).replace("{BACKEND}", backend).replace("{DIR}", d).replace("{BUDGET}", budget))
PYEOF
  T_START=$(date +%s)
  claude -p "$(cat $RUN/brief_$TAG.md)" --model $MODEL --max-turns $MAXTURNS --dangerously-skip-permissions --output-format text \
    < /dev/null > $RUN/reviewer_$TAG.log 2>&1
  for k in $(seq 1 120); do [ -f $AD/done ] && break; sleep 5; done
  T_END=$(date +%s)
  for p in $(ss -ltnp | grep ":9160 " | grep -o "pid=[0-9]*" | cut -d= -f2); do kill "$p" 2>/dev/null; done
  J=$($RD/../../scripts/agent_judge_wrap.sh $TASK $RUN/traj_$TAG.npy $AD 2>/dev/null)
  NDEC=$(ls $AD/cmds 2>/dev/null | wc -l); NOVR=$(grep -l '"override"' $AD/cmds/*.json 2>/dev/null | wc -l)
  WAITS=$(grep -o "([0-9.]*s)" $RUN/roll_$TAG.log | tr -d '()s' | sort -n | awk '{a[NR]=$1} END{print (NR?a[int((NR+1)/2)]:"")}')
  python3 - "$J" "$TAG" "$TASK" "$ARM" "$i" "$NDEC" "$NOVR" "$WAITS" "$((T_END - T_START))" >> $OUTD/${TASK}_${ARM}.jsonl <<'PYEOF'
import json, sys
j = json.loads(sys.argv[1]) if sys.argv[1].strip().startswith("{") else {"judge_error": sys.argv[1][:200]}
j.update({"tag": sys.argv[2], "task": sys.argv[3], "arm": sys.argv[4], "trial": int(sys.argv[5]), "decisions": int(sys.argv[6]),
          "overrides": int(sys.argv[7]), "median_wait_s": float(sys.argv[8]) if sys.argv[8] else None, "flight_s": int(sys.argv[9])})
print(json.dumps(j))
PYEOF
  mkdir -p $RD/agentflight/${TAG}_decisions && cp -r $AD/cmds $AD/obs $AD/*.json $RD/agentflight/${TAG}_decisions/ 2>/dev/null
  cp $RUN/traj_$TAG.npy $RD/agentflight/ 2>/dev/null; cp $RUN/reviewer_$TAG.log $RD/agentflight/${TAG}_decisions/reviewer.log 2>/dev/null
  tail -1 $OUTD/${TASK}_${ARM}.jsonl | cut -c1-240
done
