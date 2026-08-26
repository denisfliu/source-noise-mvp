"""Corrective sketch for the CMPL switch (2026-08-25). Seven coarse waypoints covering ONLY
the segment the head gets wrong: from the left-gate exit, along the +y side of the center
gate, through it southbound, ending just past the aperture — the head owns everything before
(atomic left, 10/10) and after (CFL goal hook, in-distribution from the exit point) under the
swapped prompt.

This file is the machine-derived stand-in for a HUMAN sketch: waypoints are read off the CFL
demo corridor (arc-length mean of synth3 eps 0-49) — demo-derived route knowledge, which is
rule-clean supervision. The click-UI (viz/sketchpad) replaces it with an actual hand drawing.

  python make_sketch_cmpl.py   -> sketch_cmpl.json
"""
import json
import os

RD = os.path.dirname(os.path.abspath(__file__))

SKETCH = {
    # [x, y, z, yaw] — yaw from the demo corridor (roughly path heading, swinging south)
    # starts at (1.55, 0.55): far enough east that the pre-gate approach (min dist ~0.68)
    # cannot trigger the polyline-nearest activation at enter_radius 0.5 — the first screen's
    # (1.25, 0.62) start point was reachable across the left gate frame mid-crossing
    "points": [
        [1.55, 0.55, 1.48, 0.19],
        [2.01, 0.61, 1.42, 0.07],
        [2.48, 0.53, 1.44, -0.32],
        [2.76, 0.27, 1.46, -1.01],
        [2.74, -0.16, 1.48, -1.39],  # last point before the aperture plane (y=-0.33)
        [2.77, -0.61, 1.50, -2.42],  # just past the gate: hand back to the head here
    ],
    "prompt_after": "go through the center gate from the left and hover over the stuffed animal",
    "enter_radius": 0.5,
    "step_m": 0.025,
    "sigma_serve": 0.0,
    "end_margin_m": 0.10,
}

out = f"{RD}/sketch_cmpl.json"
json.dump(SKETCH, open(out, "w"), indent=1)
print(f"wrote {out}")
