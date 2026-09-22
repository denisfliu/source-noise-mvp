"""Cloud page for the agent-in-the-loop evaluation (docs/AGENT_EVAL_PLAN.md): one viewer per task with a group per
arm, coloured by the task's own judge (experiments/rung3/agent_eval/<task>_<arm>.jsonl), over the left_and_center
cloud with the mannequin, the orbit centre and the judge marks.

  python build_agent_eval_page.py --out agent_eval.html
"""
import argparse, glob, json, os, sys
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
import cloudviewer                       # noqa: E402
from build_traj_page import marks, axes   # noqa: E402
RD = os.path.dirname(SP); AF = os.path.join(RD, "agentflight"); EV = os.path.join(RD, "agent_eval")
TASKS = [("mannequin", "Find the mannequin (14 decisions, start facing -x)"), ("double", "Left gate, then centre gate, hover over the penguin (14)"),
         ("orbit", "One full circle around the centre gate (20)"), ("left_mannequin", "Left gate, then find the mannequin (24)")]
ARMS = [("ours", [80, 220, 120]), ("waypoints", [90, 170, 240])]
OK, BAD = None, [240, 80, 80]
MANN = np.array([7.4, -0.2, 0.0], np.float32)


def fixed_marks():
    th = np.linspace(0, 2 * np.pi, 33, dtype=np.float32)
    ring = np.stack([MANN[0] + 0.5 * np.cos(th), MANN[1] + 0.5 * np.sin(th), np.full_like(th, 0.05)], 1)
    g = [{"label": "mannequin (7.4, -0.2)", "fixed": True, "color": [255, 60, 200], "trajs": [np.stack([MANN, MANN + [0, 0, 1.7]]), ring]}]
    try:
        sys.path.insert(0, RD); from agent_judges import centre_gate_xy
        c = centre_gate_xy("left_and_center")
        for r, col in ((0.8, [200, 200, 200]), (2.0, [200, 200, 200])):
            g.append({"label": f"orbit band r = {r} m about the centre gate", "fixed": True, "color": col,
                      "trajs": [np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th), np.full_like(th, 1.5)], 1)]})
    except Exception as e:
        print("no orbit marks:", e)
    return g + marks("left_and_center") + axes()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="agent_eval.html"); a = ap.parse_args()
    body = ["<title>Agent Evaluation Flights</title>", "<h2>Agent-in-the-loop evaluation: real-only pin (ours) vs waypoints, per task</h2>",
            "<p class='kv'>Green = success by the task's judge, red = failure; one group per arm. Fresh Sonnet reviewer per trial, identical briefs.</p>"]
    for task, title in TASKS:
        groups, rows = [], []
        for arm, col in ARMS:
            rec = {}
            f = os.path.join(EV, f"{task}_{arm}.jsonl")
            if os.path.exists(f):
                for l in open(f):
                    try: j = json.loads(l); rec[j["tag"]] = j
                    except Exception: pass
            ok, bad = [], []
            for tf in sorted(glob.glob(os.path.join(AF, f"traj_{task}_{arm}_t*.npy"))):
                tag = os.path.basename(tf)[5:-4]; P = np.load(tf)[:, :3].astype(np.float32); j = rec.get(tag, {})
                (ok if j.get("success") else bad).append(P)
                rows.append((arm, tag, j.get("success"), j.get("min_clearance"), j.get("decisions"), j.get("overrides"), j.get("median_wait_s")))
            if ok: groups.append({"label": f"{arm}: success ({len(ok)})", "color": col, "trajs": ok})
            if bad: groups.append({"label": f"{arm}: failure ({len(bad)})", "color": BAD if arm == "ours" else [255, 140, 40], "trajs": bad})
        if not rows:
            continue
        tab = "<table class='tt'><tr><th>arm</th><th>trial</th><th>success</th><th>min clearance</th><th>decisions</th><th>overrides</th><th>median wait s</th></tr>" + \
              "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td><td>{r[6]}</td></tr>" for r in rows) + "</table>"
        body.append(f"<h3>{title}</h3>" + cloudviewer.viewer_html("left_and_center", groups + fixed_marks(), elem_id=f"v_{task}", max_pts=40000) + tab)
    open(os.path.join(SP, a.out), "w").write("\n".join(body)); print("wrote", a.out)


main()
