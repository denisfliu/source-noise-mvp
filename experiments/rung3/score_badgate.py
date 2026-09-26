"""Flawed left/right-gate sketches (2026-09-25, run_badgate25.sh). A route is completed when the flight passes the gate
in the correct direction with no wrong-way pass (gate_success.judge) and reaches the end of the sketch without touching
the gate (route_contact). Also reports the closest approach to the gate before the route end and the tracking error.

  env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
      ~/code/openpi/.venv/bin/python score_badgate.py
"""
import glob, json, os, sys

import numpy as np
import torch

RD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, RD)
import gate_clearance as G  # noqa: E402
import gate_success as GS  # noqa: E402
import route_contact as RC  # noqa: E402

RUN = "/home/dfliu/ctxrun"
ARMS = [("Ours, sigma 0", "pin0"), ("Ours, sigma 0.5", "pin05"), ("Velocity projection on pi0", "vproj"), ("SDEdit on pi0", "sde05")]


def main():
    out = {}
    for side in ["left", "right"]:
        S = np.asarray(json.load(open(f"{RD}/sketch_bad_{side}.json"))["points"], np.float64)[:, :3]
        Q, _ = RC.dense(S)
        cloud = torch.tensor(np.asarray(G.gate_cloud(side), np.float32))
        print(f"== {side} gate: the sketch comes "
              f"{torch.cdist(torch.from_numpy(Q.astype(np.float32)), cloud).min().item():.3f} m from a post")
        for name, arm in ARMS:
            fs = sorted(glob.glob(f"{RUN}/traj_bg25_{arm}_{side}_[0-9]*.npy"))
            ok, dmins, trk, fail = 0, [], [], {"route": 0, "contact": 0, "end": 0}
            for f in fs:
                P = np.load(f)[:, :3].astype(np.float64)
                j = GS.judge(P, side)
                route = j["transit"] and j["wrong_dir_crossings"] == 0
                e, dmin = RC.contact(P, S, cloud)
                reached = e < len(P) - 1
                ok += route and reached and dmin >= RC.BODY
                fail["route"] += not route; fail["contact"] += route and dmin < RC.BODY; fail["end"] += route and not reached
                dmins.append(dmin); trk.append(np.linalg.norm(P[:e + 1, None] - Q[None], axis=2).min(1).mean())
            out[f"{arm}_{side}"] = dict(n=len(fs), completed=int(ok), closest_median_m=float(np.median(dmins)),
                                        tracking_cm=float(np.median(trk) * 100), **fail)
            print(f"   {name:28s} {ok}/{len(fs)} completed | closest to the gate median {np.median(dmins):.2f} m "
                  f"| tracking {np.median(trk) * 100:.1f} cm | route failed {fail['route']}, touched {fail['contact']}, "
                  f"did not finish {fail['end']}")
    json.dump(out, open(f"{RUN}/badgate25_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
