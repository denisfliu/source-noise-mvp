"""Trajectory judges for the agent-in-the-loop evaluation (docs/AGENT_EVAL_PLAN.md, 2026-09-22).

  python agent_judges.py --task mannequin|double|orbit|left_mannequin --traj traj_X.npy [--dir mailbox] [--scene left_and_center]

Trajectories are N x 4 (x, y, z, rollout yaw; heading = -yaw) from gate_rollout_batch.py; older N x 3 files
take the final heading from the mailbox's latest.json. The double-gate task reuses gate_success.py (compound
judge) and gate_clearance.py; the others are defined here. Prints one JSON line.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import yaml

RD = os.path.dirname(os.path.abspath(__file__))
FALSIFY = os.path.expanduser("~/code/falsify-pi")
VENVPY = "/home/dfliu/code/openpi/.venv/bin/python"
TV = "/home/dfliu/code/tv/bin/python"
EV = ["env", "-u", "VIRTUAL_ENV", "PYTHONPATH=/home/dfliu/code/openpi-snmvp/src", "JAX_PLATFORMS=cpu", "CUDA_VISIBLE_DEVICES=-1"]
MANNEQUIN = np.array([7.4, -0.2])          # RESEARCH_LOG 2026-09-21 correction, against the back wall
HOLD = 20                                   # final steps that must satisfy the hover criterion
HOVER_M = 3.0                               # reached it: within 3 m (Denis, 2026-09-23, from the videos)
HEAD_DEG = 45.0                             # looking at it: facing within 45 deg (not tested under 0.5 m)
Z_MIN, Z_MAX = 0.8, 2.0


def centre_gate_xy(scene="left_and_center"):
    saf = yaml.safe_load(open(f"{FALSIFY}/configs/safety/{scene}.yaml"))
    g = saf["ordered_miss_gate"]["gates"][-1]["corners"]
    return np.asarray(g, float)[:, :2].mean(0)


def load(traj, mailbox=None):
    P = np.load(traj)
    if P.shape[1] >= 4:
        return P[:, :3], -P[:, 3]
    head = np.full(len(P), np.nan)
    if mailbox and os.path.exists(os.path.join(mailbox, "latest.json")):
        head[:] = json.load(open(os.path.join(mailbox, "latest.json")))["pose"][3]
    return P[:, :3], head


def clearance(traj, scene):
    out = subprocess.run([TV, f"{RD}/gate_clearance.py", "--scene", scene, "--traj", traj], capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if "min-clearance" in l]
    if not line:
        return {"min_clearance": None, "clean": None}
    v = float(line[0].split("min-clearance")[1].split()[0])
    return {"min_clearance": v, "clean": "CLEAN=True" in line[0]}


def judge_gate(traj, side):
    out = subprocess.run(EV + [VENVPY, f"{RD}/gate_success.py", "--traj", traj, "--side", side], capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if l.startswith("traj_")]
    if not line:
        return {"transit": None, "route_clean": None, "goal": None, "judge_success": None, "transit_step": None}
    l = line[0]
    def field(k):
        return l.split(k + "=")[1].split()[0] if k + "=" in l else None
    if "gates=" in l:   # the ordered multi-gate judge: "gates=2/2 steps=[60, 223] dwell=159 SUCCESS=True"
        g = field("gates"); a, b = [int(x) for x in g.split("/")]
        steps = l.split("steps=[")[1].split("]")[0] if "steps=[" in l else ""
        steps = [int(x) for x in steps.replace(" ", "").split(",") if x]
        dw = field("dwell"); dw = int(dw) if dw and dw.isdigit() else 0
        return {"transit": a == b, "gates": g, "route_clean": a == b, "goal": dw > 0,
                "judge_success": field("SUCCESS") == "True", "transit_step": steps[0] if steps else None, "gate_steps": steps,
                "dwell": dw}
    tr = field("transit"); step = tr.split("@")[1] if tr and "@" in tr else None
    return {"transit": tr.startswith("True") if tr else None, "route_clean": field("wrong_dir") == "0",
            "goal": field("goal") == "True", "judge_success": field("SUCCESS") == "True",
            "transit_step": int(step) if step and step.isdigit() else None}


def mannequin(P, head, start=0):
    """Reached it and is looking at it (Denis, 2026-09-23, after reviewing the videos: within about 3 m, facing it
    within 45 deg; when the drone is right on top of the figure the facing test is meaningless, so under 0.5 m
    counts as reached). Judged on the final HOLD steps' mean pose. Clearance is combined in main and reported
    separately: a gate touch is a failure and is named."""
    seg, hs = P[start:], head[start:]
    if len(seg) < HOLD:
        return {"success": False, "reason": "too short"}
    tail, th = seg[-HOLD:], hs[-HOLD:]
    d = np.hypot(tail[:, 0] - MANNEQUIN[0], tail[:, 1] - MANNEQUIN[1])
    bearing = np.arctan2(MANNEQUIN[1] - tail[:, 1], MANNEQUIN[0] - tail[:, 0])
    off = np.degrees(np.abs((th - bearing + np.pi) % (2 * np.pi) - np.pi))
    dmean, offmean = float(d.mean()), float(np.nanmean(off))
    ok_d = dmean <= HOVER_M
    ok_h = dmean <= 0.5 or offmean <= HEAD_DEG
    ok_z = bool(seg[:, 2].min() >= Z_MIN)
    dall = np.hypot(seg[:, 0] - MANNEQUIN[0], seg[:, 1] - MANNEQUIN[1])
    return {"success": bool(ok_d and ok_h and ok_z), "final_dist": round(float(d[-1]), 3), "hold_mean_dist": round(dmean, 3),
            "hold_mean_heading_off_deg": round(offmean, 1), "hold_max_dist": round(float(d.max()), 3),
            "hold_max_heading_off_deg": round(float(np.nanmax(off)), 1), "closest": round(float(dall.min()), 3),
            "z_min": round(float(seg[:, 2].min()), 3), "reason": "" if ok_d and ok_h and ok_z else
            ("distance" if not ok_d else "heading" if not ok_h else "altitude")}


def orbit(P, centre, r_lo=0.3, r_hi=3.0):
    """A full circle around the gate without passing through it or touching it (Denis, 2026-09-22: "as long as it
    completes a circle around the gate it should be fine"). Winding = spread of the unwrapped bearing about the
    gate midpoint over the whole flight; r_lo excludes a pass through the aperture (posts at +-0.4 m), r_hi a
    detour; contact is the clearance scorer's job (combined in main)."""
    v = P[:, :2] - centre
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    r = np.hypot(v[:, 0], v[:, 1])
    wind = float(ang.max() - ang.min())
    hit = np.where(np.abs(ang - ang[0]) >= 2 * np.pi)[0]
    ok_r = bool(r.min() >= r_lo and r.max() <= r_hi)
    ok_z = bool(P[:, 2].min() >= Z_MIN and P[:, 2].max() <= Z_MAX)
    return {"success": wind >= 2 * np.pi and ok_r and ok_z, "turn_deg": round(math.degrees(wind), 1),
            "completed_step": int(hit[0]) if len(hit) else None, "r_median": round(float(np.median(r)), 2),
            "r_min": round(float(r.min()), 2), "r_max": round(float(r.max()), 2), "r_ok": ok_r, "z_ok": ok_z,
            "centre": [round(float(c), 3) for c in centre]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["mannequin", "double", "orbit", "left_mannequin"])
    ap.add_argument("--traj", required=True); ap.add_argument("--dir", default=None); ap.add_argument("--scene", default="left_and_center")
    a = ap.parse_args()
    P, head = load(a.traj, a.dir)
    out = {"task": a.task, "traj": os.path.basename(a.traj), "steps": int(len(P)),
           "path_m": round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), 2)}
    out.update(clearance(a.traj, a.scene))
    if a.task == "mannequin":
        out.update(mannequin(P, head)); out["task_done"] = bool(out["success"]); out["success"] = bool(out["success"] and out["clean"])
    elif a.task == "orbit":
        out.update(orbit(P, centre_gate_xy(a.scene))); out["task_done"] = bool(out["success"]); out["success"] = bool(out["success"] and out["clean"])
    elif a.task == "double":
        # Denis, 2026-09-23: "we just need to get through both gates without crashing" -- the hover is not required
        out.update(judge_gate(a.traj, "left_and_center")); out["task_done"] = bool(out["transit"] and out["route_clean"]); out["success"] = bool(out["task_done"] and out["clean"])
    elif a.task == "left_mannequin":
        g = judge_gate(a.traj, "left"); out.update({"left_" + k: v for k, v in g.items()})
        st = g["transit_step"] or 0
        m = mannequin(P, head, start=st); out.update({"mann_" + k: v for k, v in m.items()})
        out["task_done"] = bool(g["transit"] and g["route_clean"] and m["success"])
        out["success"] = bool(out["task_done"] and out["clean"])
    if out.get("min_clearance") is not None:
        out["gate_touch"] = bool(out["min_clearance"] < 0.18)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
