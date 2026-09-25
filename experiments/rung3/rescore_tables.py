"""Per-flight route verdicts for every paper-table cell, under whatever gate apertures the judges currently read
(2026-09-25, Denis: re-measured apertures, "show me the diff between the results"). Writes one JSON so two runs,
before and after an aperture change, can be diffed flight by flight.

  env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
      ~/code/openpi/.venv/bin/python rescore_tables.py --out ~/ctxrun/rescore_<label>.json

Table 1: success = correct-direction transit of the correct aperture, no wrong-direction crossing, and >= 0.18 m
from the gate cloud (clearance does not depend on the aperture; it is read from the stored gate_clearance scores).
Tables 2-3, relocated gates: moved_gate_cell.score_traj (crossing, no wrong way, goal after). The agent table is
re-judged separately by scripts/agent_rejudge.sh.
"""
import argparse, glob, json, os, sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, RD); sys.path.insert(0, f"{RD}/viz")
import gate_success as GS  # noqa: E402
from falsify.safety import posthoc  # noqa: E402
import moved_gate_cell as MG  # noqa: E402
from catalogue import AUTO  # noqa: E402

RUN = "/home/dfliu/ctxrun"
T1 = {"pi0 left": ("armscrreal_apc25_left", "left"), "pi0 right": ("armscrreal_apc25_right", "right"),
      "ours left": ("armrealonly_apc25_left", "left"), "ours right": ("armrealonly_apc25_right", "right")}
POSES = ["-45,0,0", "-25,0,0", "25,0,0", "45,0,0", "90,0,0", "0,0.5,-0.3", "30,-0.4,0.4",
         "100,-1.26,1.50", "180,0,0", "-35,1.34,0.75", "90,0.74,1.95", "90,-0.26,0.85"]
RELOC = {"ours": "rr25mg", "SDEdit": "sde25mg", "v-proj": "vproj25mg"}


def trajs(prefix):
    fs = glob.glob(f"{RUN}/traj_{prefix}_[0-9]*.npy")
    return sorted(fs, key=lambda f: int(f.rsplit("_", 1)[1][:-4]))


def normal_direction_judge(P, side):
    """gate_success.judge with the crossing direction taken from the motion along the aperture's normal instead of
    the per-step y-velocity (which is noise at the 48-degree left gate). The normal is oriented so that its y
    component has the task's expected sign, which keeps the +-y "from left/right" meaning."""
    scene, safety = GS.load_cfg(side)
    c, u, v, n, hu, hv = posthoc._aperture_basis(np.asarray(safety["miss_gate"]["corners"], np.float64))
    n = n * np.sign(n[1]) * GS.EXPECTED_DY_SIGN[side]
    s = (P - c) @ n
    first, wrong = None, 0
    for i in np.where(s[:-1] * s[1:] < 0)[0]:
        q = P[i] + s[i] / (s[i] - s[i + 1]) * (P[i + 1] - P[i]) - c
        if abs(q @ u) > hu or abs(q @ v) > hv:
            continue
        if s[i + 1] > s[i]:
            first = i if first is None else first
        else:
            wrong += 1
    return {"transit": first is not None, "wrong_dir_crossings": wrong}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--direction", choices=["dy", "normal"], default="dy",
                    help="dy = falsify's per-step y-velocity label (the judge as published); normal = proposed fix")
    a = ap.parse_args()
    out = {}
    for cell, (pre, side) in T1.items():
        rows = {}
        for f in trajs(pre):
            stem = os.path.basename(f)[:-4]
            P = np.load(f)[:, :3].astype(np.float64)
            j = GS.judge(P, side) if a.direction == "dy" else normal_direction_judge(P, side)
            clean = AUTO[stem]["minclr"] >= 0.18
            rows[stem] = dict(transit=j["transit"], wrong=j["wrong_dir_crossings"], clean=clean,
                              ok=bool(j["transit"] and j["wrong_dir_crossings"] == 0 and clean))
        out[f"T1 {cell}"] = rows
    for arm, pre in RELOC.items():
        rows = {}
        for i, spec in enumerate(POSES):
            dyaw, dx, dy = map(float, spec.split(","))
            tag = pre + spec.translate(str.maketrans({"-": "m", ",": "_", ".": "_"})) + f"_{41 + i}"
            for f in trajs(tag):
                r = MG.score_traj(np.load(f)[:, :3], dyaw, dx, dy)
                rows[os.path.basename(f)[:-4]] = dict(cross=r["cross"], wrong=r["wrong"], goal=r["goal"], ok=bool(r["ok"]))
        out[f"reloc {arm}"] = rows
    json.dump(out, open(a.out, "w"), indent=0)
    for k, v in out.items():
        print(f"{k:16s} {sum(r['ok'] for r in v.values())}/{len(v)}")


if __name__ == "__main__":
    main()
