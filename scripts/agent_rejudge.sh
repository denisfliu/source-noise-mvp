#!/bin/bash
# Re-judge every archived suite flight with the current agent_judges.py and rewrite the jsonl records in place.
#   bash scripts/agent_rejudge.sh [min_trial=11]
MIN=${1:-11}; cd /home/dfliu/code/source-noise-mvp
python3 - "$MIN" <<'PYEOF'
import json, glob, os, subprocess, sys
MIN=int(sys.argv[1]); RUN="/home/dfliu/ctxrun"
for p in sorted(glob.glob("experiments/rung3/agent_eval/*.jsonl")):
    rows=[json.loads(l) for l in open(p)]; changed=0
    for r in rows:
        if r["trial"]<MIN: continue
        tf=f"{RUN}/traj_{r['tag']}.npy"; d=f"{RUN}/agent_sim_{r['tag']}"
        if not os.path.exists(tf): tf=f"experiments/rung3/agentflight/traj_{r['tag']}.npy"
        if not os.path.exists(tf): continue
        out=subprocess.run(["bash","scripts/agent_judge_wrap.sh",r["task"],tf,d],capture_output=True,text=True).stdout.strip()
        if out.startswith("{"):
            j=json.loads(out); before=r.get("success"); r.update(j); changed+= (before!=r["success"])
    open(p,"w").write("\n".join(json.dumps(r) for r in rows)+"\n")
    print(os.path.basename(p), "rows", len(rows), "verdict changes", changed)
PYEOF
