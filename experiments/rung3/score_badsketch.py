"""Completed routes and tracking error for the flawed two-gate sketches (2026-09-25, run_badsketch25.sh).
A route is completed when the flight crosses both gates in order in the correct direction with no wrong-way pass
anywhere (gate_success.judge_compound, goal dwell not required) and reaches the end of the sketch without touching a
gate (route_contact). Tracking error is the mean distance to the sketch up to the route end.

  env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
      ~/code/openpi/.venv/bin/python score_badsketch.py
"""
import glob, json, os, sys

import numpy as np
import torch

RD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, RD)
import gate_clearance as G  # noqa: E402
import gate_success as GS  # noqa: E402
import route_contact as RC  # noqa: E402

RUN = "/home/dfliu/ctxrun"
CELLS = [("L", "left_and_center", "sketch_cmpl_min4.json"), ("R", "right_and_center", "sketch_cmpr_min4.json")]
ARMS = [("Ours, $\\sigma=0$", "pin0"), ("Ours, $\\sigma=0.5$", "pin05"), ("Velocity projection on $\\pi_0$", "vproj"),
        ("SDEdit on $\\pi_0$", "sde05")]


def main():
    out = {}
    for side, scene, sk in CELLS:
        S = np.asarray(json.load(open(f"{RD}/{sk}"))["points"], np.float64)[:, :3]
        Q, _ = RC.dense(S)
        cloud = torch.tensor(np.asarray(G.gate_cloud(scene), np.float32))
        sd = torch.cdist(torch.from_numpy(Q.astype(np.float32)), cloud).min(1).values.min().item()
        print(f"== {sk}: the sketch itself comes {sd:.2f} m from a gate")
        for name, arm in ARMS:
            fs = sorted(glob.glob(f"{RUN}/traj_bs25_{arm}_{side}_[0-9]*.npy"))
            ok, trk, why = 0, [], {"route": 0, "contact": 0, "end": 0}
            for f in fs:
                P = np.load(f)[:, :3].astype(np.float64)
                j = GS.judge_compound(P, scene)
                route = j["gates_latched"] == 2 and sum(j["wrong_crossings"]) == 0
                e, dmin = RC.contact(P, S, cloud)
                reached = e < len(P) - 1
                ok += route and reached and dmin >= RC.BODY
                why["route"] += not route; why["contact"] += route and dmin < RC.BODY; why["end"] += route and not reached
                trk.append(np.linalg.norm(P[:e + 1, None] - Q[None], axis=2).min(1).mean())
            out[f"{arm}_{side}"] = dict(n=len(fs), completed=ok, tracking_cm=float(np.median(trk) * 100) if trk else None, **why)
            print(f"   {name:34s} {ok}/{len(fs)} completed, tracking {np.median(trk) * 100:.1f} cm   "
                  f"(route failed {why['route']}, touched a gate {why['contact']}, did not reach the end {why['end']})")
    json.dump(out, open(f"{RUN}/badsketch25_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
