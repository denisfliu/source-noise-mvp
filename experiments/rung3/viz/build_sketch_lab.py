"""Sketch Lab (2026-09-25): draw a route on a top-down view of a scene, choose sigma and which methods to fly, and
save it to the artifact's db (collection "sketches"); Claude reads the queued sketches, flies them, and writes the
results back into the same documents, which the page lists live.

  python3 build_sketch_lab.py        (writes sketch_lab.html; publish with capabilities {"db": {}})
Plan view: +x up, +y right (Denis's convention). Gate openings come from the judge's safety YAMLs.
"""
import json, os

import numpy as np
import yaml

SP = os.path.dirname(os.path.abspath(__file__)); RD = os.path.dirname(SP)
FAL = os.path.expanduser("~/code/falsify-pi/configs/safety")
SCENES = {"left": ["left"], "right": ["right"], "left_and_center": ["left", "center"], "right_and_center": ["right", "center"]}
PER_SCENE = 12000


def gate_xy(name):
    C = yaml.safe_load(open(f"{FAL}/{name}_gate.yaml"))["miss_gate"]["corners"]
    return [[round(C[0][0], 3), round(C[0][1], 3)], [round(C[1][0], 3), round(C[1][1], 3)]]


def cloud(scene):
    Z = np.load(f"{SP}/scene_cloud_{scene}.npz"); p, c = Z["pts"], Z["rgb"]
    m = (p[:, 2] < 2.0)
    p, c = p[m], c[m]
    o = np.argsort(p[:, 2])                      # draw low points first so gates and the table sit on top
    p, c = p[o], c[o]
    if len(p) > PER_SCENE:
        k = np.sort(np.random.default_rng(0).choice(len(p), PER_SCENE, replace=False)); p, c = p[k], c[k]
    return [v for row in zip(np.round(p[:, 0], 2), np.round(p[:, 1], 2), c[:, 0], c[:, 1], c[:, 2]) for v in map(float, row)]


def preset(name, label):
    d = json.load(open(f"{RD}/sketch_{name}.json"))
    return {"label": label, "points": [[round(p[0], 3), round(p[1], 3)] for p in d["points"]], "sigma": d.get("sigma_serve", 0.0)}


def main():
    data = {"scenes": {s: cloud(s) for s in SCENES}, "sceneGates": SCENES,
            "gates": {g: gate_xy(g) for g in ["left", "right", "center"]},
            "goal": [1.525, -0.615], "start": [0.0, 0.0],
            "presets": {"left": [preset("bad_left", "Flawed left-gate sketch (6.9 cm from a post)")],
                        "right": [preset("bad_right", "Flawed right-gate sketch (6.5 cm from a post)")],
                        "left_and_center": [preset("fig8_denis3_cx15", "Figure-eight (as flown in Table 1)")],
                        "right_and_center": []}}
    html = open(f"{SP}/sketch_lab_template.html").read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    open(f"{SP}/sketch_lab.html", "w").write(html)
    print(f"wrote sketch_lab.html ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
