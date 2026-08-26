"""Authoritative gate-task success judgment — delegates to falsify.safety.posthoc.

Replaces every ad-hoc scorer in the rollout clients (north-star rule 2). Geometry
comes from the published falsify-pi configs, never from constants in scripts:
  - configs/safety/<scene>.yaml  miss_gate.corners  = the TRUE aperture rectangle
  - configs/safety/<scene>.yaml  miss_gate.goal_position + goal_tolerance_half_extents
  - configs/scenes/<scene>.yaml  gate_region        = AABB fallback / region stats

Verdict for a trajectory (MOCAP positions, (N,3)):
  transit  — >=1 aperture crossing with the correct dy sign (posthoc directional scan)
  goal     — >=1 post-transit frame inside the goal box
  success  — transit AND goal
CLI: python gate_success.py --traj t.npy [t2.npy ...] --side left|right [--json]
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

FALSIFY = os.path.expanduser("~/code/falsify-pi")
sys.path.insert(0, os.path.join(FALSIFY, "src"))
from falsify.safety import posthoc

# center-task signs measured from all 100 center demos (50/50 unanimous, 2026-08-05)
EXPECTED_DY_SIGN = {"left": +1, "right": -1, "center_from_left": -1, "center_from_right": +1}
SCENE_YAML = {"left": "left_gate", "right": "right_gate",
              "center_from_left": "center_gate", "center_from_right": "center_gate"}


def load_cfg(side):
    name = SCENE_YAML[side]
    scene = yaml.safe_load(open(f"{FALSIFY}/configs/scenes/{name}.yaml"))
    safety = yaml.safe_load(open(f"{FALSIFY}/configs/safety/{name}.yaml"))
    return scene, safety


def judge(positions, side, scene=None, safety=None):
    if scene is None or safety is None:
        scene, safety = load_cfg(side)
    mg = safety["miss_gate"]
    corners = np.asarray(mg["corners"], dtype=np.float64)
    region = scene["gate_region"]
    aabb_min = np.asarray(region["aabb_min"], dtype=np.float64)
    aabb_max = np.asarray(region["aabb_max"], dtype=np.float64)
    res = posthoc.check_directional_transit(
        np.asarray(positions, dtype=np.float64), aabb_min, aabb_max,
        expected_dy_sign=EXPECTED_DY_SIGN[side], aperture_corners=corners)
    goal = np.asarray(mg["goal_position"], dtype=np.float64)
    half = np.asarray(mg.get("goal_tolerance_half_extents") or [mg["goal_tolerance_m"]] * 3,
                      dtype=np.float64)
    in_goal = np.all(np.abs(positions - goal) <= half, axis=1)
    t0 = res.first_correct_step
    goal_after = bool(in_goal[t0 + 1:].any()) if t0 is not None else False
    return {
        "transit": t0 is not None,
        "transit_step": t0,
        "wrong_dir_crossings": res.wrong_crossings,
        "goal_after_transit": goal_after,
        # route-clean rule (2026-08-25): demos are unanimously wrong=0, so any
        # wrong-direction aperture pass is oscillation through the hoop, not a route
        "success": (t0 is not None) and goal_after and res.wrong_crossings == 0,
    }


def judge_compound(positions, scene_name="left_and_center"):
    """Ordered two-gate success (falsify ordered_miss_gate): gate_1 aperture crossing
    (correct direction) must precede gate_2's, with ZERO wrong-direction aperture
    passes anywhere in the flight; then the goal box + dwell. Geometry from
    configs/safety/<scene>.yaml.

    Route-clean rule (2026-08-25, Denis video veto of the CFG-w=4 CMPL row): a
    wrong-direction pass through an aperture means the flight reached the far side
    by threading the hoop backwards, and the later 'correct' crossing is just the
    U-turn back — the ordered latch alone cannot distinguish that from a real
    around-the-side approach, so wrong_crossings must be 0 for success."""
    safety = yaml.safe_load(open(f"{FALSIFY}/configs/safety/{scene_name}.yaml"))
    omg = safety["ordered_miss_gate"]
    P = np.asarray(positions, dtype=np.float64)
    # expected crossing directions: gate_1 like its parent task; gate_2 approached
    # from gate_1's side (left_and_center: left gate dy=+1, then center from +y side dy=-1)
    dy_signs = {"left_and_center": (+1, -1), "right_and_center": (-1, +1)}[scene_name]
    steps, wrongs = [], []
    t_prev = 0
    for g, dy in zip(omg["gates"], dy_signs):
        corners = np.asarray(g["corners"], dtype=np.float64)
        # wrong-direction passes are counted over the WHOLE flight, not just the
        # post-latch suffix — the backwards pass precedes the latched crossing
        full = posthoc.check_directional_transit(
            P, corners.min(0) - 1e-3, corners.max(0) + 1e-3,
            expected_dy_sign=dy, aperture_corners=corners)
        wrongs.append(int(full.wrong_crossings))
        res = posthoc.check_directional_transit(
            P[t_prev:], corners.min(0) - 1e-3, corners.max(0) + 1e-3,
            expected_dy_sign=dy, aperture_corners=corners)
        if res.first_correct_step is None:
            return {"gates_latched": len(steps), "success": False, "steps": steps,
                    "wrong_crossings": wrongs}
        t_prev += res.first_correct_step
        steps.append(int(t_prev))
    mg = omg
    goal = np.asarray(mg["goal_position"], dtype=np.float64)
    half = np.asarray(mg.get("goal_tolerance_half_extents", [0.3, 0.3, 0.5]), dtype=np.float64)
    ing = np.all(np.abs(P - goal) <= half, axis=1)
    best = cur = 0
    for x in ing[steps[-1] + 1:]:
        cur = cur + 1 if x else 0; best = max(best, cur)
    return {"gates_latched": 2, "steps": steps, "goal_dwell": int(best),
            "wrong_crossings": wrongs, "route_clean": sum(wrongs) == 0,
            "success": best >= 16 and sum(wrongs) == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--side", required=True,
                    choices=list(EXPECTED_DY_SIGN) + ["left_and_center", "right_and_center"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.side in ("left_and_center", "right_and_center"):
        n_success = 0
        for t in a.traj:
            P = np.load(t)[:, :3]
            v = judge_compound(P, a.side)
            n_success += v["success"]
            print("%-28s gates=%d/2 steps=%s dwell=%s  SUCCESS=%s" % (
                os.path.basename(t), v["gates_latched"], v.get("steps"),
                v.get("goal_dwell"), v["success"]))
        print("== %d/%d success (%s, ordered 2-gate + dwell)" % (n_success, len(a.traj), a.side))
        return
    scene, safety = load_cfg(a.side)
    n_success = 0
    for t in a.traj:
        P = np.load(t)[:, :3]
        v = judge(P, a.side, scene, safety)
        n_success += v["success"]
        line = {"traj": os.path.basename(t), **v}
        print(json.dumps(line) if a.json else
              "%-28s transit=%s@%s wrong_dir=%d goal=%s  SUCCESS=%s" %
              (os.path.basename(t), v["transit"], v["transit_step"],
               v["wrong_dir_crossings"], v["goal_after_transit"], v["success"]))
    print("== %d/%d success (%s)" % (n_success, len(a.traj), a.side))


if __name__ == "__main__":
    main()
