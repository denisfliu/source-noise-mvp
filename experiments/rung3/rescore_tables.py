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
import moved_gate_cell as MG  # noqa: E402
from catalogue import AUTO  # noqa: E402

RUN = "/home/dfliu/ctxrun"
T1 = {"pi0 left": ("armscrreal_apc25_left", "left"), "pi0 right": ("armscrreal_apc25_right", "right"),
      "ours left": ("armrealonly_apc25_left", "left"), "ours right": ("armrealonly_apc25_right", "right")}
# (GATE_TF spec, seed): random far-away poses, 2026-09-25 (>= 1.2 m from takeoff, >= 1.0 m from the goal, >= 0.8 m between
# gate centres, sketch >= 0.25 m from the gate); the tag is the spec with "-" -> "m" and "," "." -> "_", then _seed
POSES = [("-163,0.04,-1.09", 51), ("174,2.81,1.37", 52), ("2,1.06,2.68", 53), ("153,2.46,-0.3", 54), ("-58,1.55,1.76", 55),
         ("132,1.69,-0.83", 56), ("-49,-0.06,2.97", 57), ("44,2.03,2.6", 58), ("130,0.42,-0.36", 59), ("-86,2.1,0.87", 60)]
RELOC = {"ours": "rr25mg", "SDEdit": "sde25mg", "v-proj": "vproj25mg"}


def trajs(prefix):
    fs = glob.glob(f"{RUN}/traj_{prefix}_[0-9]*.npy")
    return sorted(fs, key=lambda f: int(f.rsplit("_", 1)[1][:-4]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = {}
    for cell, (pre, side) in T1.items():
        rows = {}
        for f in trajs(pre):
            stem = os.path.basename(f)[:-4]
            P = np.load(f)[:, :3].astype(np.float64)
            j = GS.judge(P, side)
            clean = AUTO[stem]["minclr"] >= 0.18
            rows[stem] = dict(transit=j["transit"], wrong=j["wrong_dir_crossings"], clean=clean,
                              ok=bool(j["transit"] and j["wrong_dir_crossings"] == 0 and clean))
        out[f"T1 {cell}"] = rows
    for arm, pre in RELOC.items():
        rows = {}
        for spec, seed in POSES:
            dyaw, dx, dy = map(float, spec.split(","))
            tag = pre + spec.translate(str.maketrans({"-": "m", ",": "_", ".": "_"})) + f"_{seed}"
            for f in trajs(tag):
                r = MG.score_traj(np.load(f)[:, :3], dyaw, dx, dy)
                rows[os.path.basename(f)[:-4]] = dict(cross=r["cross"], wrong=r["wrong"], goal=r["goal"], ok=bool(r["ok"]))
        out[f"reloc {arm}"] = rows
    json.dump(out, open(a.out, "w"), indent=0)
    for k, v in out.items():
        print(f"{k:16s} {sum(r['ok'] for r in v.values())}/{len(v)}")


if __name__ == "__main__":
    main()
